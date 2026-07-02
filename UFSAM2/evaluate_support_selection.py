import argparse
import json
import os
import random
from os.path import join
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import opts
from datasets import build_dataset
from models.sansa.sansa import build_sansa
from train_uncertainty_head import IoUHead
from util.commons import make_deterministic, resume_from_checkpoint, setup_logging
from util.promptable_utils import build_prompt_dict


def _to_python(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.cpu().tolist()
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return _to_python(value[0])
        return [_to_python(v) for v in value]
    return value


def _binary_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> float:
    pred_mask = pred_mask.bool()
    gt_mask = gt_mask.bool()
    inter = torch.logical_and(pred_mask, gt_mask).sum().float()
    union = torch.logical_or(pred_mask, gt_mask).sum().float()
    if union.item() == 0:
        return 1.0
    return (inter / union).item()


def _trace_tensor(trace: dict[str, Any], key: str) -> torch.Tensor:
    value = trace[key]
    if value is None:
        raise ValueError(f"Missing trace tensor `{key}`.")
    return value.reshape(-1).float()


def load_uncertainty_head(ckpt_path: str, device: torch.device) -> tuple[IoUHead, tuple[str, ...], torch.Tensor, torch.Tensor]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    feature_keys = tuple(ckpt["feature_keys"])
    mean = ckpt["mean"].float()
    std = ckpt["std"].float().clamp_min(1e-6)
    hidden_dim = int(ckpt.get("args", {}).get("hidden_dim", ckpt["model"]["net.0.weight"].shape[0]))
    dropout = float(ckpt.get("args", {}).get("dropout", 0.0))
    head = IoUHead(mean.numel(), hidden_dim, dropout).to(device)
    head.load_state_dict(ckpt["model"])
    head.eval()
    return head, feature_keys, mean.to(device), std.to(device)


@torch.no_grad()
def predict_expected_iou(
    head: IoUHead,
    feature_keys: tuple[str, ...],
    mean: torch.Tensor,
    std: torch.Tensor,
    trace: dict[str, Any],
    device: torch.device,
) -> float:
    parts = [_trace_tensor(trace, key) for key in feature_keys]
    feature = torch.cat(parts).to(device)
    feature = (feature - mean) / std
    return head(feature.unsqueeze(0)).item()


@torch.no_grad()
def run_sansa_episode(
    model: torch.nn.Module,
    support_imgs: torch.Tensor,
    support_masks: torch.Tensor,
    query_img: torch.Tensor,
    query_mask: torch.Tensor,
    support_indices: list[int],
    args: argparse.Namespace,
) -> tuple[float, float, torch.Tensor, dict[str, Any]]:
    selected_imgs = support_imgs[0, support_indices]
    selected_masks = support_masks[:, support_indices]
    imgs = torch.cat([selected_imgs, query_img]).unsqueeze(0)
    img_h, img_w = imgs.shape[-2:]

    imgs = imgs.to(args.device)
    prompt_dict = build_prompt_dict(
        selected_masks,
        args.prompt,
        n_shots=len(support_indices),
        train_mode=False,
        device=model.device,
    )
    outputs = model(imgs, prompt_dict, return_traces=True)
    pred_masks = outputs["pred_masks"].unsqueeze(0)
    pred_masks = F.interpolate(pred_masks, size=(img_h, img_w), mode="bilinear", align_corners=False)
    pred_query = (pred_masks.sigmoid() > args.threshold)[0, -1].cpu()
    true_iou = _binary_iou(pred_query, query_mask[0].cpu())
    trace = outputs["traces"][-1]
    sam_score = trace["sam_score"].flatten().max().item()
    return true_iou, sam_score, pred_query, trace


def mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ["random", "sam_score", "token", "oracle", "all_supports"]
    summary = {"episodes": len(records)}
    for key in keys:
        values = [record["ious"][key] for record in records]
        summary[f"miou_{key}"] = mean(values)
    summary["token_oracle_match"] = mean([float(record["selected"]["token"] == record["selected"]["oracle"]) for record in records])
    summary["sam_score_oracle_match"] = mean([float(record["selected"]["sam_score"] == record["selected"]["oracle"]) for record in records])
    summary["token_beats_sam_score"] = mean([float(record["ious"]["token"] > record["ious"]["sam_score"]) for record in records])
    return summary


def save_results(path: str, records: list[dict[str, Any]], args: argparse.Namespace) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "summary": summarize(records),
        "records": records,
        "args": vars(args),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def evaluate(model: torch.nn.Module, args: argparse.Namespace) -> list[dict[str, Any]]:
    validation_ds = "coco" if args.dataset_file == "multi" else args.dataset_file
    ds = build_dataset(validation_ds, image_set="val", args=args)
    dataloader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers)
    head_device = torch.device(args.head_device)
    head, feature_keys, mean_vec, std_vec = load_uncertainty_head(args.head_ckpt, head_device)

    model.eval()
    rng = random.Random(args.seed)
    max_episodes = len(dataloader) if args.max_episodes is None else min(args.max_episodes, len(dataloader))
    records = []
    pbar = tqdm(dataloader, total=max_episodes, desc="support-select", ncols=100)

    for episode_idx, batch in enumerate(pbar):
        if episode_idx >= max_episodes:
            break

        query_img, query_mask = batch["query_img"], batch["query_mask"]
        support_imgs, support_masks = batch["support_imgs"], batch["support_masks"]
        n_supports = support_imgs.shape[1]

        per_support = []
        for support_idx in range(n_supports):
            true_iou, sam_score, _, trace = run_sansa_episode(
                model,
                support_imgs,
                support_masks,
                query_img,
                query_mask,
                [support_idx],
                args,
            )
            token_score = predict_expected_iou(head, feature_keys, mean_vec, std_vec, trace, head_device)
            per_support.append(
                {
                    "support_idx": support_idx,
                    "true_iou": true_iou,
                    "sam_score": sam_score,
                    "token_score": token_score,
                }
            )

        all_iou, all_sam_score, _, all_trace = run_sansa_episode(
            model,
            support_imgs,
            support_masks,
            query_img,
            query_mask,
            list(range(n_supports)),
            args,
        )
        all_token_score = predict_expected_iou(head, feature_keys, mean_vec, std_vec, all_trace, head_device)

        random_idx = rng.randrange(n_supports)
        sam_idx = max(per_support, key=lambda item: item["sam_score"])["support_idx"]
        token_idx = max(per_support, key=lambda item: item["token_score"])["support_idx"]
        oracle_idx = max(per_support, key=lambda item: item["true_iou"])["support_idx"]
        iou_by_idx = {item["support_idx"]: item["true_iou"] for item in per_support}

        record = {
            "episode_idx": episode_idx,
            "dataset": validation_ds,
            "class_id": _to_python(batch.get("class_id")),
            "category": _to_python(batch.get("category")),
            "query_name": _to_python(batch.get("query_name")),
            "support_names": _to_python(batch.get("support_names")),
            "per_support": per_support,
            "selected": {
                "random": random_idx,
                "sam_score": sam_idx,
                "token": token_idx,
                "oracle": oracle_idx,
            },
            "ious": {
                "random": iou_by_idx[random_idx],
                "sam_score": iou_by_idx[sam_idx],
                "token": iou_by_idx[token_idx],
                "oracle": iou_by_idx[oracle_idx],
                "all_supports": all_iou,
            },
            "all_supports": {
                "sam_score": all_sam_score,
                "token_score": all_token_score,
            },
        }
        records.append(record)

        if args.save_every > 0 and len(records) % args.save_every == 0:
            save_results(args.output_path, records, args)

        summary = summarize(records)
        pbar.set_postfix(
            random=f"{summary['miou_random']:.3f}",
            sam=f"{summary['miou_sam_score']:.3f}",
            token=f"{summary['miou_token']:.3f}",
            oracle=f"{summary['miou_oracle']:.3f}",
            all=f"{summary['miou_all_supports']:.3f}",
        )

    return records


def main(args: argparse.Namespace) -> None:
    if args.prompt != "mask":
        raise ValueError("Support selection evaluation is currently mask-only.")
    if args.shots < 2:
        raise ValueError("Support selection needs --shots >= 2.")

    setup_logging(args.output_dir, console="info", rank=0)
    make_deterministic(args.seed)
    model = build_sansa(args.sam2_version, args.adaptformer_stages, args.channel_factor, args.device)
    model.to(torch.device(args.device))
    if args.resume:
        resume_from_checkpoint(args.resume, model)

    records = evaluate(model, args)
    save_results(args.output_path, records, args)
    print(json.dumps(summarize(records), indent=2))
    print(f"Saved support-selection results to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluate 5-shot support selection with SANSA uncertainty.", parents=[opts.get_args_parser()])
    parser.add_argument("--head_ckpt", type=str, required=True)
    parser.add_argument("--head_device", type=str, default="cpu")
    parser.add_argument("--output_path", type=str, default="output/support_selection.json")
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--save_every", type=int, default=20)
    args = parser.parse_args()
    args.output_dir = join(args.output_dir, args.name_exp)
    main(args)
