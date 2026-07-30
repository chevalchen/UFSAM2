"""Train the single frozen EXP-002 Stage A mask refiner."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess

import torch
from torch.utils.data import DataLoader

from models.sansa.mask_post_correction import mask_post_refiner_state_dict
from util.commons import make_deterministic, setup_logging
from util.episode_manifest import file_sha256
from util.losses import loss_masks
from util.mask_post_correction_runtime import (
    build_model_with_official_adapter,
    freeze_except_mask_post_refiner,
    load_stage_a_contract,
    replay_partition_from_contract,
    resolve_repository_path,
)
from util.promptable_utils import build_prompt_dict


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _prepare_batch(
    batch: dict[str, torch.Tensor],
    *,
    shots: int,
    prompt: str,
    device: torch.device,
    model_device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    query_img = batch["query_img"]
    query_mask = batch["query_mask"]
    support_imgs = batch["support_imgs"]
    support_masks = batch["support_masks"]
    samples = torch.cat(
        (support_imgs[:, :shots], query_img.unsqueeze(1)),
        dim=1,
    ).to(device)
    masks = torch.cat(
        (support_masks[:, :shots], query_mask.unsqueeze(1)),
        dim=1,
    )
    prompt_dict = build_prompt_dict(
        support_masks[:, :shots],
        prompt,
        n_shots=shots,
        train_mode=True,
        device=model_device,
    )
    return samples, masks, prompt_dict


def main(args: argparse.Namespace) -> None:
    contract = load_stage_a_contract(args.contract)
    training = contract["training"]
    formal_steps = int(training["max_optimizer_steps"])
    requested_steps = (
        int(args.diagnostic_max_steps)
        if args.diagnostic_only
        else formal_steps
    )
    if args.diagnostic_only and requested_steps <= 0:
        raise ValueError("--diagnostic_max_steps must be positive.")
    if not args.diagnostic_only and args.diagnostic_max_steps is not None:
        raise ValueError("--diagnostic_max_steps requires --diagnostic_only.")

    run_dir = Path(args.output_dir) / args.name_exp
    if run_dir.exists():
        unexpected = [
            path.name for path in run_dir.iterdir() if path.name != "run_manifest.json"
        ]
        if unexpected:
            raise FileExistsError(
                f"Refusing to overwrite non-empty run directory {run_dir}: {unexpected}"
            )
    setup_logging(str(run_dir), console="info", rank=0)
    make_deterministic(int(training["seed"]))

    model, adapter_keys = build_model_with_official_adapter(
        contract,
        adapter_path=args.adapter_checkpoint,
        sam2_checkpoint_path=args.sam2_checkpoint,
        device=args.device,
    )
    trainable = freeze_except_mask_post_refiner(model)
    model.eval()
    assert model.mask_post_refiner is not None
    model.mask_post_refiner.train()

    train_dataset = replay_partition_from_contract(
        contract,
        "train",
        data_root=args.data_root,
    )
    loader = DataLoader(
        train_dataset,
        batch_size=int(training["batch_size"]),
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=True,
    )
    optimizer_config = training["optimizer"]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(optimizer_config["lr"]),
        weight_decay=float(optimizer_config["weight_decay"]),
        betas=tuple(float(value) for value in optimizer_config["betas"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=formal_steps,
    )
    device = torch.device(args.device)
    shots = int(contract["dataset"]["shots"])
    prompt = str(contract["dataset"]["prompt"])
    epochs = int(training["epochs"])
    global_step = 0
    epoch_summaries: list[dict[str, float | int]] = []

    for epoch in range(epochs):
        loss_sum = 0.0
        steps_this_epoch = 0
        for batch in loader:
            samples, masks, prompt_dict = _prepare_batch(
                batch,
                shots=shots,
                prompt=prompt,
                device=device,
                model_device=model.device,
            )
            outputs = model(samples, prompt_dict)
            losses = loss_masks(outputs["pred_masks"], masks, num_frames=1)
            loss = sum(losses.values())
            loss_value = float(loss.detach().cpu())
            if not math.isfinite(loss_value):
                raise RuntimeError(f"Non-finite training loss at step {global_step}: {loss_value}")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable,
                float(training["clip_max_norm"]),
            )
            optimizer.step()
            scheduler.step()

            global_step += 1
            steps_this_epoch += 1
            loss_sum += loss_value
            if global_step % 100 == 0 or global_step == requested_steps:
                print(
                    json.dumps(
                        {
                            "event": "train_step",
                            "epoch": epoch + 1,
                            "global_step": global_step,
                            "loss": loss_value,
                            "grad_norm": float(grad_norm),
                            "lr": optimizer.param_groups[0]["lr"],
                        }
                    )
                )
            if global_step >= requested_steps:
                break

        epoch_summaries.append(
            {
                "epoch": epoch + 1,
                "steps": steps_this_epoch,
                "mean_loss": loss_sum / max(steps_this_epoch, 1),
            }
        )
        if global_step >= requested_steps:
            break

    if not args.diagnostic_only:
        if global_step != formal_steps or len(epoch_summaries) != epochs:
            raise RuntimeError(
                "Formal training exposure mismatch: "
                f"steps={global_step}/{formal_steps}, epochs={len(epoch_summaries)}/{epochs}."
            )

    contract_path = resolve_repository_path(args.contract)
    manifest_path = resolve_repository_path(
        contract["dataset"]["episode_manifest"]["path"]
    )
    adapter_path = Path(
        args.adapter_checkpoint
        or contract["baseline"]["official_adapter"]["path"]
    )
    checkpoint_name = (
        "mask_post_refiner_diagnostic.pth"
        if args.diagnostic_only
        else "mask_post_refiner_epoch10_final.pth"
    )
    checkpoint_path = run_dir / checkpoint_name
    torch.save(
        {
            "schema_version": 1,
            "experiment_id": "EXP-002",
            "stage": "operator_headroom",
            "evidence_label": (
                "DIAGNOSTIC_ONLY" if args.diagnostic_only else "FORMAL_TRAINING_ARTIFACT"
            ),
            "git_sha": _git_sha(),
            "model": mask_post_refiner_state_dict(model),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": scheduler.state_dict(),
            "epochs_completed": len(epoch_summaries),
            "global_step": global_step,
            "contract_sha256": file_sha256(contract_path),
            "episode_manifest_sha256": file_sha256(manifest_path),
            "adapter_sha256": file_sha256(adapter_path),
        },
        checkpoint_path,
    )
    summary = {
        "schema_version": 1,
        "experiment_id": "EXP-002",
        "stage": "operator_headroom",
        "evidence_label": (
            "DIAGNOSTIC_ONLY" if args.diagnostic_only else "FORMAL_TRAINING_ARTIFACT"
        ),
        "git_sha": _git_sha(),
        "global_step": global_step,
        "epochs_completed": len(epoch_summaries),
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "loaded_adapter_tensors": len(adapter_keys),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "epoch_summaries": epoch_summaries,
        "validation_executed": False,
    }
    (run_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train EXP-002 Stage A fixed mask refiner")
    parser.add_argument(
        "--contract",
        default="experiments/EXP-002/configs/stage_a_mask_refiner_contract.json",
    )
    parser.add_argument("--adapter_checkpoint", default=None)
    parser.add_argument("--sam2_checkpoint", default=None)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--name_exp", required=True)
    parser.add_argument("--diagnostic_only", action="store_true")
    parser.add_argument("--diagnostic_max_steps", type=int, default=None)
    main(parser.parse_args())
