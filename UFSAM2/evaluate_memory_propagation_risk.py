import argparse
import json
import os
from os.path import join
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm

import opts
from datasets import build_dataset
from evaluate_support_selection import load_uncertainty_head, predict_expected_iou
from models.sansa.sansa import build_sansa
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


def _extract_score(trace: dict[str, Any]) -> float:
    value = trace.get("sam_score")
    if value is None:
        return float("nan")
    return value.flatten().max().item()


@torch.no_grad()
def run_episode(
    model: torch.nn.Module,
    support_imgs: torch.Tensor,
    support_masks: torch.Tensor,
    query_imgs: list[torch.Tensor],
    query_masks: list[torch.Tensor],
    args: argparse.Namespace,
) -> tuple[list[float], list[dict[str, Any]]]:
    imgs = torch.cat([support_imgs, torch.stack(query_imgs)]).unsqueeze(0)
    img_h, img_w = imgs.shape[-2:]
    prompt_dict = build_prompt_dict(
        support_masks.unsqueeze(0),
        args.prompt,
        n_shots=args.shots,
        train_mode=False,
        device=model.device,
    )

    outputs = model(imgs.to(args.device), prompt_dict, return_traces=True)
    pred_masks = outputs["pred_masks"].unsqueeze(0)
    pred_masks = F.interpolate(pred_masks, size=(img_h, img_w), mode="bilinear", align_corners=False)
    pred_queries = (pred_masks.sigmoid() > args.threshold)[0, args.shots :].cpu()
    ious = [_binary_iou(pred_queries[i], query_masks[i].cpu()) for i in range(len(query_imgs))]
    return ious, outputs["traces"]


@torch.no_grad()
def run_independent_queries(
    model: torch.nn.Module,
    support_imgs: torch.Tensor,
    support_masks: torch.Tensor,
    query_imgs: list[torch.Tensor],
    query_masks: list[torch.Tensor],
    args: argparse.Namespace,
) -> tuple[list[float], list[dict[str, Any]]]:
    ious = []
    traces = []
    for query_img, query_mask in zip(query_imgs, query_masks):
        query_ious, query_traces = run_episode(
            model,
            support_imgs,
            support_masks,
            [query_img],
            [query_mask],
            args,
        )
        ious.append(query_ious[0])
        traces.append(query_traces[0])
    return ious, traces


def _sample_chain(ds: Any, class_idx: int, chain_length: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    support_item = ds[class_idx]
    query_items = [support_item]
    for _ in range(chain_length - 1):
        query_items.append(ds[class_idx])
    return support_item, query_items


def _mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    x = torch.tensor(xs, dtype=torch.float32)
    y = torch.tensor(ys, dtype=torch.float32)
    if x.std(unbiased=False).item() == 0 or y.std(unbiased=False).item() == 0:
        return None
    return torch.corrcoef(torch.stack([x, y]))[0, 1].item()


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"chains": 0}

    chain_length = len(records[0]["queries"])
    summary: dict[str, Any] = {"chains": len(records), "chain_length": chain_length}

    for pos in range(chain_length):
        indep = [record["queries"][pos]["independent_iou"] for record in records]
        seq = [record["queries"][pos]["sequential_iou"] for record in records]
        harm = [record["queries"][pos]["harm"] for record in records]
        summary[f"q{pos + 1}_independent_miou"] = _mean(indep)
        summary[f"q{pos + 1}_sequential_miou"] = _mean(seq)
        summary[f"q{pos + 1}_mean_harm"] = _mean(harm)
        summary[f"q{pos + 1}_harm_gt_0.01"] = _mean([float(v > 0.01) for v in harm])
        summary[f"q{pos + 1}_harm_gt_0.05"] = _mean([float(v > 0.05) for v in harm])

    all_indep = [query["independent_iou"] for record in records for query in record["queries"]]
    all_seq = [query["sequential_iou"] for record in records for query in record["queries"]]
    all_harm = [query["harm"] for record in records for query in record["queries"]]
    downstream_harm = [query["harm"] for record in records for query in record["queries"][1:]]
    summary["all_independent_miou"] = _mean(all_indep)
    summary["all_sequential_miou"] = _mean(all_seq)
    summary["all_mean_harm"] = _mean(all_harm)
    summary["downstream_mean_harm"] = _mean(downstream_harm)
    summary["downstream_harm_gt_0.01"] = _mean([float(v > 0.01) for v in downstream_harm])
    summary["downstream_harm_gt_0.05"] = _mean([float(v > 0.05) for v in downstream_harm])

    prev_scores = []
    next_harms = []
    prev_ious = []
    for record in records:
        for pos in range(chain_length - 1):
            prev = record["queries"][pos]
            nxt = record["queries"][pos + 1]
            if prev.get("sequential_token_score") is not None:
                prev_scores.append(prev["sequential_token_score"])
                next_harms.append(nxt["harm"])
            prev_ious.append(prev["sequential_iou"])
    summary["prev_token_score_vs_next_harm_pearson"] = _pearson(prev_scores, next_harms)
    summary["prev_seq_iou_vs_next_harm_pearson"] = _pearson(prev_ious, next_harms)
    return summary


def save_results(path: str, records: list[dict[str, Any]], args: argparse.Namespace) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {"summary": summarize(records), "records": records, "args": vars(args)}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def evaluate(model: torch.nn.Module, args: argparse.Namespace) -> list[dict[str, Any]]:
    validation_ds = "coco" if args.dataset_file == "multi" else args.dataset_file
    if validation_ds == "paco_part":
        raise ValueError("This probe needs class-indexed sampling; PACO-Part needs a dedicated sampler.")
    ds = build_dataset(validation_ds, image_set="val", args=args)

    head = None
    feature_keys = ()
    mean_vec = std_vec = None
    head_device = torch.device(args.head_device)
    if args.head_ckpt:
        head, feature_keys, mean_vec, std_vec = load_uncertainty_head(args.head_ckpt, head_device)

    records = []
    model.eval()
    n_classes = len(getattr(ds, "class_ids", range(len(ds))))
    max_chains = args.max_chains if args.max_chains is not None else n_classes
    pbar = tqdm(range(max_chains), desc="memory-risk", ncols=100)

    for chain_idx in pbar:
        class_idx = chain_idx % n_classes
        support_item, query_items = _sample_chain(ds, class_idx, args.chain_length)
        support_imgs = support_item["support_imgs"][: args.shots]
        support_masks = support_item["support_masks"][: args.shots]
        query_imgs = [item["query_img"] for item in query_items]
        query_masks = [item["query_mask"] for item in query_items]

        independent_ious, independent_traces = run_independent_queries(
            model,
            support_imgs,
            support_masks,
            query_imgs,
            query_masks,
            args,
        )
        sequential_ious, sequential_traces = run_episode(
            model,
            support_imgs,
            support_masks,
            query_imgs,
            query_masks,
            args,
        )

        queries = []
        for pos in range(args.chain_length):
            independent_score = None
            sequential_score = None
            if head is not None and mean_vec is not None and std_vec is not None:
                independent_score = predict_expected_iou(
                    head, feature_keys, mean_vec, std_vec, independent_traces[pos], head_device
                )
                sequential_score = predict_expected_iou(
                    head, feature_keys, mean_vec, std_vec, sequential_traces[pos], head_device
                )
            queries.append(
                {
                    "pos": pos + 1,
                    "query_name": _to_python(query_items[pos].get("query_name")),
                    "independent_iou": independent_ious[pos],
                    "sequential_iou": sequential_ious[pos],
                    "harm": independent_ious[pos] - sequential_ious[pos],
                    "independent_sam_score": _extract_score(independent_traces[pos]),
                    "sequential_sam_score": _extract_score(sequential_traces[pos]),
                    "independent_token_score": independent_score,
                    "sequential_token_score": sequential_score,
                }
            )

        record = {
            "chain_idx": chain_idx,
            "dataset": validation_ds,
            "class_idx": class_idx,
            "class_id": _to_python(support_item.get("class_id")),
            "category": _to_python(support_item.get("category")),
            "support_names": _to_python(support_item.get("support_names")),
            "queries": queries,
        }
        records.append(record)

        if args.save_every > 0 and len(records) % args.save_every == 0:
            save_results(args.output_path, records, args)
        summary = summarize(records)
        pbar.set_postfix(
            indep=f"{summary['all_independent_miou']:.3f}",
            seq=f"{summary['all_sequential_miou']:.3f}",
            harm=f"{summary['downstream_mean_harm']:.3f}",
        )

    return records


def main(args: argparse.Namespace) -> None:
    if args.prompt != "mask":
        raise ValueError("Memory propagation risk probe is currently mask-only.")
    if args.chain_length < 2:
        raise ValueError("Use --chain_length >= 2 to measure propagation risk.")

    setup_logging(args.output_dir, console="info", rank=0)
    make_deterministic(args.seed)
    model = build_sansa(args.sam2_version, args.adaptformer_stages, args.channel_factor, args.device)
    model.to(torch.device(args.device))
    if args.resume:
        resume_from_checkpoint(args.resume, model)

    records = evaluate(model, args)
    save_results(args.output_path, records, args)
    print(json.dumps(summarize(records), indent=2))
    print(f"Saved memory propagation risk results to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Probe test-time pseudo-query memory propagation risk.", parents=[opts.get_args_parser()])
    parser.add_argument("--chain_length", type=int, default=3)
    parser.add_argument("--max_chains", type=int, default=100)
    parser.add_argument("--output_path", type=str, default="output/memory_propagation_risk.json")
    parser.add_argument("--save_every", type=int, default=20)
    parser.add_argument("--head_ckpt", type=str, default=None)
    parser.add_argument("--head_device", type=str, default="cpu")
    args = parser.parse_args()
    args.output_dir = join(args.output_dir, args.name_exp)
    main(args)
