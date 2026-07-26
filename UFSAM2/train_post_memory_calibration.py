import argparse
import os
from os.path import join

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import opts
from datasets import build_dataset
from inference_fss import eval_fss
from models.sansa.post_memory_calibration import load_post_memory_calibrator_checkpoint
from models.sansa.sansa import build_sansa
from util.commons import make_deterministic, setup_logging
from util.checkpoint_validation import validate_sansa_base_state
from util.episode_manifest import (
    EpisodeReplayDataset,
    file_sha256,
    load_episode_manifest,
)
from util.promptable_utils import build_prompt_dict


def _load_base_checkpoint(model: nn.Module, checkpoint_path: str) -> dict:
    """Load a SANSA full-model or adapter-only checkpoint safely.

    Official SANSA training checkpoints contain only adapter parameters. Missing
    frozen SAM2 parameters are therefore expected, while missing configured
    adapter parameters, unknown keys, shape mismatches, or pre-existing AV-PMC
    parameters are hard errors.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    if not isinstance(state, dict):
        raise ValueError(f"Unsupported SANSA checkpoint format: {checkpoint_path}")

    model_state = model.state_dict()
    loadable_state, expected_adapter_keys = validate_sansa_base_state(
        model_state,
        state,
    )
    incompatible = model.load_state_dict(loadable_state, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(
            f"Unexpected keys after SANSA checkpoint load: {incompatible.unexpected_keys}"
        )
    loaded_adapter_keys = sorted(
        key
        for key in loadable_state
        if "adapter" in key and key not in incompatible.missing_keys
    )
    if loaded_adapter_keys != expected_adapter_keys:
        raise RuntimeError(
            "Not every configured SANSA adapter parameter was loaded. "
            f"Loaded {len(loaded_adapter_keys)}/{len(expected_adapter_keys)}."
        )
    print(
        f"Loaded {len(loaded_adapter_keys)} SANSA adapter tensors from "
        f"{checkpoint_path}; frozen SAM2 parameters came from the configured base weights."
    )
    return checkpoint


def _as_binary_mask(mask: torch.Tensor, size: tuple[int, int], device: torch.device) -> torch.Tensor:
    mask = mask.to(device=device, dtype=torch.float32)
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    elif mask.ndim == 5 and mask.size(1) == 1:
        mask = mask[:, 0]
    if mask.ndim != 4:
        raise ValueError(f"Expected query mask as BHW or B1HW, got {tuple(mask.shape)}")
    if mask.size(1) != 1:
        mask = mask[:, :1]
    return F.interpolate(mask, size=size, mode="nearest").clamp(0.0, 1.0)


def _segmentation_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probability = logits.sigmoid()
    intersection = (probability * target).sum(dim=(-2, -1))
    denominator = probability.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
    dice = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    return bce + dice


def _binary_iou(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = logits > 0
    truth = target > 0.5
    intersection = (prediction & truth).sum(dim=(-2, -1)).float()
    union = (prediction | truth).sum(dim=(-2, -1)).float()
    return torch.where(union > 0, intersection / union.clamp_min(1.0), torch.ones_like(union))


def _episode_inputs(batch: dict, args: argparse.Namespace, model: nn.Module) -> tuple[torch.Tensor, dict]:
    support_imgs = batch["support_imgs"].to(args.device)
    query_img = batch["query_img"].to(args.device)
    if query_img.ndim == 4:
        query_img = query_img.unsqueeze(1)
    if support_imgs.ndim != 5 or query_img.ndim != 5:
        raise ValueError(
            f"Expected support BSCWH and query B1CWH tensors, got {support_imgs.shape} and {query_img.shape}"
        )
    images = torch.cat((support_imgs, query_img), dim=1)
    prompt_dict = build_prompt_dict(
        batch["support_masks"],
        args.prompt,
        n_shots=args.shots,
        train_mode=True,
        device=model.device,
    )
    return images, prompt_dict


def _stage_loss(
    stage: str,
    outputs: dict,
    query_mask: torch.Tensor,
    device: torch.device,
    sign_weight: float,
    gate_temperature: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    records = outputs.get("post_memory_calibration", [])
    if len(records) != 1:
        raise RuntimeError(
            "The staged AV-PMC trainer currently requires batch_size=1 and one query per episode."
        )
    record = records[0]
    final_logits = record["calibrated_low_res_masks"]
    baseline_logits = record["baseline_low_res_masks"]
    if final_logits is None or baseline_logits is None:
        raise RuntimeError("AV-PMC training requires both baseline and calibrated logits.")
    target = _as_binary_mask(query_mask, final_logits.shape[-2:], device)

    if stage == "operator":
        loss = _segmentation_loss(final_logits, target)
        return loss, {"seg": float(loss.detach().item())}

    if stage == "spatial":
        baseline_pixel_loss = F.binary_cross_entropy_with_logits(
            baseline_logits.detach(), target, reduction="none"
        )
        repaired_pixel_loss = F.binary_cross_entropy_with_logits(
            final_logits.detach(), target, reduction="none"
        )
        dense_target = (baseline_pixel_loss - repaired_pixel_loss).clamp(-1.0, 1.0)
        score = record["spatial_benefit"]
        dense_target = F.interpolate(dense_target, size=score.shape[-2:], mode="area")
        regression = F.smooth_l1_loss(score, dense_target)
        sign = F.binary_cross_entropy_with_logits(
            score / gate_temperature,
            (dense_target > 0).to(score.dtype),
        )
        loss = regression + sign_weight * sign
        return loss, {
            "dense_reg": float(regression.detach().item()),
            "dense_sign": float(sign.detach().item()),
            "positive_area": float((dense_target > 0).float().mean().item()),
        }

    if stage == "gain":
        baseline_iou = _binary_iou(baseline_logits.detach(), target)
        repaired_iou = _binary_iou(final_logits.detach(), target)
        delta_iou = (repaired_iou - baseline_iou).mean(dim=1, keepdim=True)
        predicted_delta = record["predicted_delta_iou"]
        loss = F.smooth_l1_loss(predicted_delta, delta_iou)
        return loss, {
            "gain_loss": float(loss.detach().item()),
            "true_delta_iou": float(delta_iou.detach().mean().item()),
            "pred_delta_iou": float(predicted_delta.detach().mean().item()),
        }

    raise ValueError(f"Unknown AV-PMC training stage: {stage}")


def main(args: argparse.Namespace) -> None:
    if args.pmc_train_stage is None:
        raise ValueError("--pmc_train_stage is required: operator, spatial, or gain.")
    if not args.resume:
        raise ValueError("A verified SANSA --resume checkpoint is required for AV-PMC training.")
    if args.batch_size != 1:
        raise ValueError("Use --batch_size 1 for the staged AV-PMC trainer.")
    if args.pmc_train_stage in {"spatial", "gain"} and not args.pmc_checkpoint:
        raise ValueError("The spatial and gain stages require the preceding --pmc_checkpoint.")
    if not args.episode_manifest:
        raise ValueError(
            "--episode_manifest is required. Random, untracked episodes are not "
            "permitted for EXP-001 training."
        )

    setup_logging(args.output_dir, console="info", rank=0)
    make_deterministic(args.seed)
    training_mode = {"operator": "operator", "spatial": "operator", "gain": "spatial"}[
        args.pmc_train_stage
    ]
    validation_mode = {"operator": "operator", "spatial": "spatial", "gain": "gated"}[
        args.pmc_train_stage
    ]
    args.pmc_mode = training_mode
    model = build_sansa(
        args.sam2_version,
        args.adaptformer_stages,
        args.channel_factor,
        args.device,
        post_memory_calibration=True,
        pmc_mode=training_mode,
        pmc_projection_dim=args.pmc_projection_dim,
        pmc_hidden_dim=args.pmc_hidden_dim,
        pmc_residual_scale=args.pmc_residual_scale,
        pmc_spatial_threshold=args.pmc_spatial_threshold,
        pmc_episode_threshold=args.pmc_episode_threshold,
        pmc_gate_temperature=args.pmc_gate_temperature,
        pmc_train_stage=args.pmc_train_stage,
    )
    _load_base_checkpoint(model, args.resume)
    if args.pmc_checkpoint:
        load_post_memory_calibrator_checkpoint(
            model.post_memory_calibrator,
            args.pmc_checkpoint,
            strict=True,
        )
        model.set_post_memory_calibration_train_stage(args.pmc_train_stage)

    device = torch.device(args.device)
    model.to(device)
    model.eval()
    model.post_memory_calibrator.train()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError(f"No trainable parameters for AV-PMC stage {args.pmc_train_stage}.")
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    manifest = load_episode_manifest(args.episode_manifest)
    base_dataset = build_dataset(
        args.dataset_file,
        image_set=manifest["source_split"],
        args=args,
    )
    dataset = EpisodeReplayDataset(
        base_dataset,
        manifest,
        args.pmc_train_partition,
        data_root=args.data_root,
        expected_dataset=args.dataset_file,
        expected_fold=args.fold,
        expected_shots=args.shots,
        expected_source_split=manifest["source_split"],
    )
    validation_dataset = EpisodeReplayDataset(
        base_dataset,
        manifest,
        args.pmc_validation_partition,
        data_root=args.data_root,
        expected_dataset=args.dataset_file,
        expected_fold=args.fold,
        expected_shots=args.shots,
        expected_source_split=manifest["source_split"],
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers,
    )
    os.makedirs(args.output_dir, exist_ok=True)
    manifest_sha256 = file_sha256(args.episode_manifest)
    base_checkpoint_sha256 = file_sha256(args.resume)
    preceding_pmc_sha256 = (
        file_sha256(args.pmc_checkpoint) if args.pmc_checkpoint else None
    )

    global_step = 0
    best_validation_miou = float("-inf")
    for epoch in range(args.epochs):
        running_loss = 0.0
        epoch_steps = 0
        progress = tqdm(loader, desc=f"AV-PMC {args.pmc_train_stage} epoch {epoch + 1}")
        for batch in progress:
            images, prompt_dict = _episode_inputs(batch, args, model)
            outputs = model(
                images,
                prompt_dict,
                return_calibration_data=True,
            )
            loss, diagnostics = _stage_loss(
                args.pmc_train_stage,
                outputs,
                batch["query_mask"],
                device,
                args.pmc_dense_sign_weight,
                args.pmc_gate_temperature,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(trainable, args.clip_max_norm)
            optimizer.step()

            global_step += 1
            epoch_steps += 1
            running_loss += float(loss.detach().item())
            progress.set_postfix(loss=f"{running_loss / epoch_steps:.4f}", **diagnostics)
            if args.pmc_max_steps is not None and global_step >= args.pmc_max_steps:
                break

        args.episode_partition = args.pmc_validation_partition
        model.post_memory_calibrator.set_mode(validation_mode)
        args.pmc_mode = validation_mode
        validation_metrics_path = join(
            args.output_dir,
            f"pmc_{args.pmc_train_stage}_validation_epoch{epoch + 1}.json",
        )
        validation_summary = eval_fss(
            model,
            args,
            dataset=validation_dataset,
            metrics_path=validation_metrics_path,
            provenance={
                "model_state": f"in_memory_after_epoch_{epoch + 1}",
                "base_checkpoint": args.resume,
                "base_checkpoint_sha256": base_checkpoint_sha256,
                "preceding_pmc_checkpoint": args.pmc_checkpoint or None,
                "preceding_pmc_checkpoint_sha256": preceding_pmc_sha256,
                "episode_manifest_sha256": manifest_sha256,
            },
        )
        model.eval()
        model.post_memory_calibrator.set_mode(training_mode)
        model.post_memory_calibrator.train()
        args.pmc_mode = training_mode
        validation_miou = float(validation_summary["treatment_miou"])
        is_best = validation_miou > best_validation_miou
        best_validation_miou = max(best_validation_miou, validation_miou)

        checkpoint_path = join(
            args.output_dir,
            f"pmc_{args.pmc_train_stage}_epoch{epoch + 1}.pth",
        )
        checkpoint_payload = {
            "post_memory_calibrator": model.post_memory_calibrator.state_dict(),
            "stage": args.pmc_train_stage,
            "epoch": epoch,
            "global_step": global_step,
            "args": vars(args),
            "base_checkpoint_sha256": base_checkpoint_sha256,
            "preceding_pmc_checkpoint_sha256": preceding_pmc_sha256,
            "episode_manifest_sha256": manifest_sha256,
            "episode_manifest_partition": args.pmc_train_partition,
            "validation_partition": args.pmc_validation_partition,
            "training_mode": training_mode,
            "validation_mode": validation_mode,
            "validation_summary": validation_summary,
            "best_validation_miou": best_validation_miou,
        }
        torch.save(checkpoint_payload, checkpoint_path)
        print(f"Saved {checkpoint_path}")
        if is_best:
            best_path = join(
                args.output_dir,
                f"pmc_{args.pmc_train_stage}_best.pth",
            )
            torch.save(checkpoint_payload, best_path)
            print(
                f"Saved best {args.pmc_train_stage} checkpoint to {best_path} "
                f"(base-validation mIoU={validation_miou:.2f})"
            )
        if args.pmc_max_steps is not None and global_step >= args.pmc_max_steps:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Train AV-PMC in operator, spatial-benefit, or action-value stages",
        parents=[opts.get_args_parser()],
    )
    parser.add_argument("--pmc_image_set", type=str, default="train", help="Deprecated compatibility alias; the manifest source_split is authoritative.")
    parser.add_argument("--pmc_train_partition", type=str, default="train", choices=["train"], help="Manifest partition used for optimization.")
    parser.add_argument("--pmc_validation_partition", type=str, default="validation", choices=["validation"], help="Held-out manifest partition used for best-checkpoint selection.")
    parser.add_argument("--pmc_dense_sign_weight", type=float, default=0.25, help="Weight of positive dense-benefit sign supervision.")
    parser.add_argument("--pmc_max_steps", type=int, default=None, help="Optional smoke-test cap on optimizer steps.")
    args = parser.parse_args()
    args.output_dir = join(args.output_dir, args.name_exp)
    main(args)
