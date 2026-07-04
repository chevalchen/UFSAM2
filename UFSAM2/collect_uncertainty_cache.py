import argparse
import os
import sys
from os.path import join
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import opts
import util.misc as utils
from datasets import build_dataset
from models.sansa.sansa import build_sansa
from util.commons import make_deterministic, resume_from_checkpoint, setup_logging
from util.promptable_utils import build_prompt_dict


def _to_python(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.cpu()
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return _to_python(value[0])
        return [_to_python(v) for v in value]
    return value


def _tensor_or_none(trace: dict[str, Any], key: str) -> torch.Tensor | None:
    value = trace.get(key)
    if value is None:
        return None
    return value.squeeze(0).cpu()


def _aggregate_trace_tensor(traces: list[dict[str, Any]], key: str) -> torch.Tensor | None:
    values = [_tensor_or_none(trace, key) for trace in traces]
    values = [value for value in values if value is not None]
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return torch.stack(values).mean(dim=0)


def _binary_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> float:
    pred_mask = pred_mask.bool()
    gt_mask = gt_mask.bool()
    inter = torch.logical_and(pred_mask, gt_mask).sum().float()
    union = torch.logical_or(pred_mask, gt_mask).sum().float()
    if union.item() == 0:
        return 1.0
    return (inter / union).item()


def collect_cache(model: torch.nn.Module, args: argparse.Namespace) -> list[dict[str, Any]]:
    validation_ds = "coco" if args.dataset_file == "multi" else args.dataset_file
    ds = build_dataset(validation_ds, image_set=args.image_set, args=args)
    dataloader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers)

    model.eval()
    records = []
    iou_sum = 0.0
    last_saved = 0
    max_episodes = len(dataloader) if args.max_episodes is None else min(args.max_episodes, len(dataloader))
    pbar = tqdm(
        dataloader,
        total=max_episodes,
        ncols=80,
        desc=f"{validation_ds}-{args.shots}shot",
        disable=(utils.get_rank() != 0),
        file=sys.stderr,
        dynamic_ncols=True,
    )

    for episode_idx, batch in enumerate(pbar):
        if episode_idx >= max_episodes:
            break

        query_img, query_mask = batch["query_img"], batch["query_mask"]
        support_imgs, support_masks = batch["support_imgs"], batch["support_masks"]
        imgs = torch.cat([support_imgs[0], query_img]).unsqueeze(0)
        img_h, img_w = imgs.shape[-2:]

        imgs = imgs.to(args.device)
        prompt_dict = build_prompt_dict(
            support_masks,
            args.prompt,
            n_shots=args.shots,
            train_mode=False,
            device=model.device,
        )

        with torch.no_grad():
            outputs = model(imgs, prompt_dict, return_traces=True)

        pred_masks = outputs["pred_masks"].unsqueeze(0)
        pred_masks = F.interpolate(pred_masks, size=(img_h, img_w), mode="bilinear", align_corners=False)
        pred_query = (pred_masks.sigmoid() > args.threshold)[0, -1].cpu()
        gt_query = query_mask[0].cpu()
        trace = outputs["traces"][-1]
        support_traces = outputs.get("support_traces", [])

        sam_scores = _tensor_or_none(trace, "sam_score")
        sam_score = None if sam_scores is None else sam_scores.flatten().max().item()
        support_areas = support_masks[0].flatten(1).float().sum(dim=1).cpu()

        true_iou = _binary_iou(pred_query, gt_query)
        iou_sum += true_iou

        record = {
            "dataset": validation_ds,
            "fold": args.fold,
            "shot": args.shots,
            "episode_idx": episode_idx,
            "class_id": _to_python(batch.get("class_id")),
            "category": _to_python(batch.get("category")),
            "query_name": _to_python(batch.get("query_name")),
            "support_names": _to_python(batch.get("support_names")),
            "true_iou": true_iou,
            "sam_score": sam_score,
            "query_iou_token": _tensor_or_none(trace, "query_iou_token"),
            "query_mask_token": _tensor_or_none(trace, "query_mask_token"),
            "query_mask_tokens": _tensor_or_none(trace, "query_mask_tokens"),
            "query_obj_ptr": _tensor_or_none(trace, "query_obj_ptr"),
            "query_memory_summary": _tensor_or_none(trace, "query_memory_summary"),
            "support_iou_token": _aggregate_trace_tensor(support_traces, "support_iou_token"),
            "support_mask_token": _aggregate_trace_tensor(support_traces, "support_mask_token"),
            "support_mask_tokens": _aggregate_trace_tensor(support_traces, "support_mask_tokens"),
            "support_obj_ptr": _aggregate_trace_tensor(support_traces, "support_obj_ptr"),
            "support_memory_summary": _aggregate_trace_tensor(support_traces, "support_memory_summary"),
            "pred_area": pred_query.float().sum().item(),
            "gt_area": gt_query.float().sum().item(),
            "support_area_mean": support_areas.mean().item(),
            "support_area_std": support_areas.std(unbiased=False).item(),
        }
        records.append(record)

        if args.save_every > 0 and len(records) % args.save_every == 0:
            torch.save(records, args.cache_path)
            last_saved = len(records)

        pbar.set_postfix(
            iou=f"{true_iou:.3f}",
            avg_iou=f"{iou_sum / len(records):.3f}",
            cached=len(records),
            saved=last_saved,
        )

    return records


def main(args: argparse.Namespace) -> None:
    if args.prompt != "mask":
        raise ValueError("Stage-1 uncertainty cache collection is currently mask-only.")

    setup_logging(args.output_dir, console="info", rank=0)
    make_deterministic(args.seed)
    os.makedirs(os.path.dirname(args.cache_path) or ".", exist_ok=True)

    model = build_sansa(args.sam2_version, args.adaptformer_stages, args.channel_factor, args.device)
    model.to(torch.device(args.device))
    if args.resume:
        resume_from_checkpoint(args.resume, model)

    records = collect_cache(model, args)
    torch.save(records, args.cache_path)
    print(f"Saved {len(records)} uncertainty records to {args.cache_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("SANSA uncertainty cache collector", parents=[opts.get_args_parser()])
    parser.add_argument("--cache_path", type=str, default="output/uncertainty_cache.pt")
    parser.add_argument("--image_set", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--save_every", type=int, default=100)
    args = parser.parse_args()
    args.output_dir = join(args.output_dir, args.name_exp)
    main(args)
