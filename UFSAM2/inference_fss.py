import argparse
import json
import os
import sys
from os.path import join
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import opts
from models.sansa.sansa import build_sansa
from models.sansa.post_memory_calibration import load_post_memory_calibrator_checkpoint
from datasets import build_dataset
from util.commons import make_deterministic, setup_logging, resume_from_checkpoint
from util.episode_manifest import (
    EpisodeReplayDataset,
    file_sha256,
    load_episode_manifest,
)
import util.misc as utils
from util.promptable_utils import build_prompt_dict
from util.metrics import AverageMeter, Evaluator


def main(args: argparse.Namespace) -> float:
    setup_logging(args.output_dir, console="info", rank=0)
    make_deterministic(args.seed)
    print(args)

    if args.post_memory_calibration and not args.pmc_checkpoint:
        raise ValueError("--pmc_checkpoint is required when AV-PMC is enabled.")
    model = build_sansa(
        args.sam2_version,
        args.adaptformer_stages,
        args.channel_factor,
        args.device,
        post_memory_calibration=args.post_memory_calibration,
        pmc_mode=args.pmc_mode,
        pmc_projection_dim=args.pmc_projection_dim,
        pmc_hidden_dim=args.pmc_hidden_dim,
        pmc_residual_scale=args.pmc_residual_scale,
        pmc_spatial_threshold=args.pmc_spatial_threshold,
        pmc_episode_threshold=args.pmc_episode_threshold,
        pmc_gate_temperature=args.pmc_gate_temperature,
    )
    device = torch.device(args.device)
    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if args.resume:
        resume_from_checkpoint(args.resume, model)
    if args.post_memory_calibration:
        load_post_memory_calibrator_checkpoint(
            model.post_memory_calibrator,
            args.pmc_checkpoint,
            strict=True,
        )

    print(f"number of params: {n_parameters}")
    print('Start inference')

    summary = eval_fss(model, args)
    return summary["treatment_miou"]


def eval_fss(
    model: torch.nn.Module,
    args: argparse.Namespace,
    dataset=None,
    metrics_path: str | None = None,
    provenance: dict | None = None,
) -> dict:
    """
    Evaluate SANSA on the few-shot segmentation benchmark.
    Computes and prints mIoU across the validation set.
    """
    # load data
    validation_ds = 'coco' if args.dataset_file == 'multi' else args.dataset_file
    print(f'Evaluating {validation_ds} - fold: {args.fold}')
    if dataset is None:
        if args.episode_manifest:
            manifest = load_episode_manifest(args.episode_manifest)
            base_dataset = build_dataset(
                validation_ds,
                image_set=manifest["source_split"],
                args=args,
            )
            ds = EpisodeReplayDataset(
                base_dataset,
                manifest,
                args.episode_partition,
                data_root=args.data_root,
                expected_dataset=validation_ds,
                expected_fold=args.fold,
                expected_shots=args.shots,
                expected_source_split=manifest["source_split"],
            )
        else:
            ds = build_dataset(validation_ds, image_set='val', args=args)
    else:
        ds = dataset
    dataloader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers)

    pmc_enabled = getattr(model, "post_memory_calibrator", None) is not None
    model.eval()
    average_meter = AverageMeter(args.dataset_file, ds.class_ids, ds.nclass)
    baseline_meter = (
        AverageMeter(args.dataset_file, ds.class_ids, ds.nclass)
        if pmc_enabled
        else None
    )
    oracle_meter = (
        AverageMeter(args.dataset_file, ds.class_ids, ds.nclass)
        if pmc_enabled
        else None
    )
    paired_metrics = []

    pbar = tqdm(dataloader, ncols=80, desc='runn avg.', disable=(utils.get_rank() != 0), file=sys.stderr, dynamic_ncols=True)
    for idx, batch in enumerate(pbar):
        query_img, query_mask = batch['query_img'], batch['query_mask']
        support_imgs, support_masks = batch['support_imgs'], batch['support_masks']

        imgs = torch.cat([support_imgs[0], query_img]).unsqueeze(0) # b t c h w
        img_h, img_w = imgs.shape[-2:]

        imgs = imgs.to(args.device)
        prompt_dict = build_prompt_dict(support_masks, args.prompt, n_shots=args.shots, train_mode=False, device=model.device)

        with torch.no_grad():
            outputs = model(
                imgs,
                prompt_dict,
                return_calibration_data=pmc_enabled,
            )

        pred_masks = outputs["pred_masks"].unsqueeze(0)  # [1, T, h, w]
        pred_masks = F.interpolate(pred_masks, size=(img_h, img_w), mode='bilinear', align_corners=False) 
        pred_masks = (pred_masks.sigmoid() > args.threshold)[0].cpu()

        area_inter, area_union = Evaluator.classify_prediction(pred_masks[-1:].float(), batch, device=imgs.device)
        average_meter.update(area_inter, area_union, batch['class_id'].cuda())

        if pmc_enabled:
            records = outputs.get("post_memory_calibration", [])
            if len(records) != 1:
                raise RuntimeError(
                    "EXP-001 evaluation requires exactly one query calibration record per episode."
                )
            record = records[0]
            baseline_logits = record["baseline_low_res_masks"]
            baseline_masks = F.interpolate(
                baseline_logits,
                size=(img_h, img_w),
                mode="bilinear",
                align_corners=False,
            )
            baseline_masks = (baseline_masks.sigmoid() > args.threshold)[:, 0].cpu()
            baseline_inter, baseline_union = Evaluator.classify_prediction(
                baseline_masks.float(),
                batch,
                device=imgs.device,
            )
            baseline_meter.update(
                baseline_inter,
                baseline_union,
                batch["class_id"].cuda(),
            )
            baseline_fg_iou = float(
                (baseline_inter[1].sum() / baseline_union[1].sum().clamp_min(1.0)).item()
            )
            treatment_fg_iou = float(
                (area_inter[1].sum() / area_union[1].sum().clamp_min(1.0)).item()
            )
            if treatment_fg_iou > baseline_fg_iou:
                oracle_inter, oracle_union = area_inter, area_union
            else:
                oracle_inter, oracle_union = baseline_inter, baseline_union
            oracle_meter.update(
                oracle_inter,
                oracle_union,
                batch["class_id"].cuda(),
            )
            class_id = batch["class_id"].reshape(-1)[0].item()
            episode_id = batch.get("episode_id", [None])
            if isinstance(episode_id, (list, tuple)):
                episode_id = episode_id[0]
            paired_metrics.append(
                {
                    "episode_idx": idx,
                    "episode_id": episode_id,
                    "class_id": int(class_id),
                    "baseline_iou": baseline_fg_iou,
                    "treatment_iou": treatment_fg_iou,
                    "delta_iou": treatment_fg_iou - baseline_fg_iou,
                    "predicted_delta_iou": float(
                        record["predicted_delta_iou"].detach().mean().item()
                    ),
                    "applied": bool(record["applied"]),
                    "spatial_gate_mean": float(
                        record["spatial_gate"].detach().mean().item()
                    ),
                }
            )

        if (idx + 1) % 50 == 0:
            miou, _, _ = average_meter.compute_iou()
            pbar.set_description(f"Runn. Avg mIoU = {miou:.1f}")

        if args.visualize:
            from util.visualization import visualize_episode
            fg_inter = area_inter[1].sum().item()
            fg_union = area_union[1].sum().item()
            vis_iou = fg_inter / max(fg_union, 1e-6)
            visualize_episode(
                support_imgs=[support_imgs[0, i].cpu() for i in range(args.shots)],
                query_img=query_img[0].cpu(),
                query_gt=(query_mask[0].numpy() > 0),
                query_pred=pred_masks[-1].numpy(),
                prompt_dict=prompt_dict,
                out_dir=args.output_dir,
                idx=idx,
                src_size=model.sam.image_size,
                # iou=area_inter/area_union,
                iou=vis_iou,
            )
    average_meter.write_result(args.dataset_file)
    miou, fb_iou, _ = average_meter.compute_iou()
    print('Fold %d mIoU: %5.2f \t FB-IoU: %5.2f' % (args.fold, miou, fb_iou.item()))
    if baseline_meter is not None:
        baseline_miou, baseline_fb_iou, _ = baseline_meter.compute_iou()
        oracle_miou, oracle_fb_iou, _ = oracle_meter.compute_iou()
        print(
            'Matched B0 Fold %d mIoU: %5.2f \t FB-IoU: %5.2f'
            % (args.fold, baseline_miou, baseline_fb_iou.item())
        )
        print(
            'Oracle B8 Fold %d mIoU: %5.2f \t FB-IoU: %5.2f'
            % (args.fold, oracle_miou, oracle_fb_iou.item())
        )
        os.makedirs(args.output_dir, exist_ok=True)
        if metrics_path is None:
            metrics_path = join(args.output_dir, args.pmc_metrics_file)
        metrics_payload = {
            "schema_version": 1,
            "experiment_id": "EXP-001",
            "dataset": args.dataset_file,
            "fold": args.fold,
            "shots": args.shots,
            "seed": args.seed,
            "pmc_mode": args.pmc_mode,
            "pmc_checkpoint": args.pmc_checkpoint,
            "episode_manifest": args.episode_manifest,
            "episode_partition": args.episode_partition,
            "provenance": provenance
            or {
                "base_checkpoint": args.resume,
                "base_checkpoint_sha256": (
                    file_sha256(args.resume) if args.resume else None
                ),
                "pmc_checkpoint_sha256": (
                    file_sha256(args.pmc_checkpoint)
                    if args.pmc_checkpoint
                    else None
                ),
                "episode_manifest_sha256": (
                    file_sha256(args.episode_manifest)
                    if args.episode_manifest
                    else None
                ),
            },
            "summary": {
                "baseline_b0_miou": baseline_miou,
                "treatment_miou": miou,
                "oracle_b8_miou": oracle_miou,
                "baseline_b0_fb_iou": float(baseline_fb_iou.item()),
                "treatment_fb_iou": float(fb_iou.item()),
                "oracle_b8_fb_iou": float(oracle_fb_iou.item()),
            },
            "episodes": paired_metrics,
        }
        with open(metrics_path, "w", encoding="utf-8") as handle:
            json.dump(metrics_payload, handle, ensure_ascii=False, indent=2)
        print(f"Wrote paired EXP-001 episode metrics to {metrics_path}")
    print('==================== Finished Testing ====================')

    return {
        "treatment_miou": miou,
        "treatment_fb_iou": float(fb_iou.item()),
        "baseline_miou": baseline_miou if baseline_meter is not None else None,
        "baseline_fb_iou": (
            float(baseline_fb_iou.item()) if baseline_meter is not None else None
        ),
        "oracle_miou": oracle_miou if oracle_meter is not None else None,
        "oracle_fb_iou": (
            float(oracle_fb_iou.item()) if oracle_meter is not None else None
        ),
        "metrics_path": metrics_path if baseline_meter is not None else None,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SANSA evaluation script', parents=[opts.get_args_parser()])
    args = parser.parse_args()
    args.output_dir = join(args.output_dir, args.name_exp)
    main(args)
