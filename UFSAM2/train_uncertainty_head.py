import argparse
import json
import math
import os
import random
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


FEATURE_KEYS = {
    "tokens": ("query_iou_token", "query_mask_token", "query_obj_ptr"),
    "tokens_mem": ("query_iou_token", "query_mask_token", "query_obj_ptr", "query_memory_summary"),
    "all_mask_tokens": ("query_iou_token", "query_mask_tokens", "query_obj_ptr"),
}


class UncertaintyCacheDataset(Dataset):
    def __init__(
        self,
        records: list[dict[str, Any]],
        feature_keys: tuple[str, ...],
        mean: torch.Tensor | None = None,
        std: torch.Tensor | None = None,
    ) -> None:
        self.records = records
        self.feature_keys = feature_keys
        self.features = torch.stack([self._make_feature(record) for record in records]).float()
        self.targets = torch.tensor([record["true_iou"] for record in records]).float()
        self.mean = self.features.mean(dim=0) if mean is None else mean
        self.std = self.features.std(dim=0, unbiased=False) if std is None else std
        self.std = self.std.clamp_min(1e-6)
        self.features = (self.features - self.mean) / self.std

    def _make_feature(self, record: dict[str, Any]) -> torch.Tensor:
        parts = []
        for key in self.feature_keys:
            value = record[key]
            if value is None:
                raise ValueError(f"Missing feature `{key}` in cache record.")
            parts.append(value.reshape(-1).float())
        return torch.cat(parts)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]


class IoUHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1).sigmoid()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _record_class_id(record: dict[str, Any]) -> Any:
    class_id = record.get("class_id")
    if isinstance(class_id, torch.Tensor):
        return class_id.item()
    return class_id


def split_records_random(
    records: list[dict[str, Any]], seed: int, train_ratio: float, val_ratio: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    indices = list(range(len(records)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    n_train = int(len(indices) * train_ratio)
    n_val = int(len(indices) * val_ratio)
    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]
    return (
        [records[i] for i in train_idx],
        [records[i] for i in val_idx],
        [records[i] for i in test_idx],
    )


def split_records_by_class(
    records: list[dict[str, Any]], seed: int, train_ratio: float, val_ratio: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    class_ids = sorted({_record_class_id(record) for record in records})
    rng = random.Random(seed)
    rng.shuffle(class_ids)
    n_train = int(len(class_ids) * train_ratio)
    n_val = int(len(class_ids) * val_ratio)
    train_classes = set(class_ids[:n_train])
    val_classes = set(class_ids[n_train : n_train + n_val])
    test_classes = set(class_ids[n_train + n_val :])
    return (
        [record for record in records if _record_class_id(record) in train_classes],
        [record for record in records if _record_class_id(record) in val_classes],
        [record for record in records if _record_class_id(record) in test_classes],
    )


def split_records(
    records: list[dict[str, Any]],
    split_by: str,
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if split_by == "random":
        return split_records_random(records, seed, train_ratio, val_ratio)
    if split_by == "class_id":
        return split_records_by_class(records, seed, train_ratio, val_ratio)
    raise ValueError(f"Unsupported split_by: {split_by}")


def split_summary(records: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "episodes": len(records),
        "classes": len({_record_class_id(record) for record in records}),
    }


def pearson(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred = pred.float()
    target = target.float()
    vx = pred - pred.mean()
    vy = target - target.mean()
    denom = vx.norm() * vy.norm()
    if denom.item() == 0:
        return float("nan")
    return (vx @ vy / denom).item()


def rankdata(x: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(x)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(len(x), dtype=torch.float32, device=x.device)
    return ranks


def spearman(pred: torch.Tensor, target: torch.Tensor) -> float:
    return pearson(rankdata(pred), rankdata(target))


def auroc(labels: torch.Tensor, scores: torch.Tensor) -> float:
    labels = labels.bool()
    n_pos = labels.sum().item()
    n_neg = (~labels).sum().item()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    pos_rank_sum = ranks[labels].sum().item()
    return (pos_rank_sum - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg)


def ece(pred: torch.Tensor, target: torch.Tensor, n_bins: int) -> float:
    total = len(pred)
    value = 0.0
    edges = torch.linspace(0, 1, n_bins + 1, device=pred.device)
    for idx in range(n_bins):
        lo, hi = edges[idx], edges[idx + 1]
        if idx == n_bins - 1:
            mask = (pred >= lo) & (pred <= hi)
        else:
            mask = (pred >= lo) & (pred < hi)
        if mask.any():
            value += mask.float().mean().item() * abs(pred[mask].mean().item() - target[mask].mean().item())
    return value if total > 0 else float("nan")


def risk_coverage(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    order = torch.argsort(pred, descending=True)
    out = {}
    for coverage in (0.5, 0.7, 0.9, 1.0):
        k = max(1, math.ceil(len(order) * coverage))
        chosen = target[order[:k]]
        out[f"risk@cov{coverage:.1f}"] = (1.0 - chosen.mean()).item()
        out[f"miou@cov{coverage:.1f}"] = chosen.mean().item()
    return out


def score_metrics(pred: torch.Tensor, target: torch.Tensor, n_bins: int) -> dict[str, float]:
    risk_score = 1.0 - pred
    metrics = {
        "mae": torch.mean(torch.abs(pred - target)).item(),
        "rmse": torch.sqrt(torch.mean((pred - target) ** 2)).item(),
        "pearson": pearson(pred, target),
        "spearman": spearman(pred, target),
        "auroc_iou_lt_0.5": auroc(target < 0.5, risk_score),
        "auroc_iou_lt_0.7": auroc(target < 0.7, risk_score),
        "ece": ece(pred, target, n_bins),
        "target_mean": target.mean().item(),
        "pred_mean": pred.mean().item(),
        "n": len(target),
    }
    metrics.update(risk_coverage(pred, target))
    return metrics


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, n_bins: int) -> dict[str, float]:
    model.eval()
    preds, targets = [], []
    for features, target in loader:
        features = features.to(device)
        preds.append(model(features).cpu())
        targets.append(target.cpu())
    pred = torch.cat(preds)
    target = torch.cat(targets)
    return score_metrics(pred, target, n_bins)


def evaluate_sam_score(records: list[dict[str, Any]], n_bins: int) -> dict[str, float]:
    target = torch.tensor([record["true_iou"] for record in records]).float()
    pred = torch.tensor([record["sam_score"] for record in records]).float()
    return score_metrics(pred, target, n_bins)


def train(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    device = torch.device(args.device)
    records = torch.load(args.cache_path, map_location="cpu", weights_only=False)
    if args.max_records is not None:
        records = records[: args.max_records]

    feature_keys = FEATURE_KEYS[args.feature_set]
    train_records, val_records, test_records = split_records(
        records, args.split_by, args.seed, args.train_ratio, args.val_ratio
    )
    if not train_records or not val_records or not test_records:
        raise ValueError(
            "Empty split after splitting records. "
            f"split_by={args.split_by}, sizes="
            f"{len(train_records)}/{len(val_records)}/{len(test_records)}. "
            "Use a larger cache or adjust --train_ratio/--val_ratio."
        )
    if args.eval_sam_score_only:
        final = {
            "sam_score": {
                "train": evaluate_sam_score(train_records, args.ece_bins),
                "val": evaluate_sam_score(val_records, args.ece_bins),
                "test": evaluate_sam_score(test_records, args.ece_bins),
            },
            "split": {
                "split_by": args.split_by,
                "train": split_summary(train_records),
                "val": split_summary(val_records),
                "test": split_summary(test_records),
            },
        }
        os.makedirs(args.output_dir, exist_ok=True)
        metrics_path = os.path.join(args.output_dir, "sam_score_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(final, f, indent=2)
        return {"checkpoint": None, "metrics": metrics_path, "final": final}

    train_ds = UncertaintyCacheDataset(train_records, feature_keys)
    val_ds = UncertaintyCacheDataset(val_records, feature_keys, train_ds.mean, train_ds.std)
    test_ds = UncertaintyCacheDataset(test_records, feature_keys, train_ds.mean, train_ds.std)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = IoUHead(train_ds.features.shape[1], args.hidden_dim, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.SmoothL1Loss(beta=args.smooth_l1_beta)

    best_val = float("inf")
    best_state = None
    pbar = tqdm(range(args.epochs), desc="train-iou-head", ncols=100)
    for epoch in pbar:
        model.train()
        loss_sum = 0.0
        seen = 0
        for features, target in train_loader:
            features = features.to(device)
            target = target.to(device)
            pred = model(features)
            loss = loss_fn(pred, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(target)
            seen += len(target)

        val_metrics = evaluate(model, val_loader, device, args.ece_bins)
        train_loss = loss_sum / max(1, seen)
        pbar.set_postfix(train_loss=f"{train_loss:.4f}", val_mae=f"{val_metrics['mae']:.4f}", val_r=f"{val_metrics['pearson']:.3f}")
        if val_metrics["mae"] < best_val:
            best_val = val_metrics["mae"]
            best_state = {
                "model": model.state_dict(),
                "mean": train_ds.mean,
                "std": train_ds.std,
                "feature_keys": feature_keys,
                "args": vars(args),
                "val_metrics": val_metrics,
            }

    if best_state is not None:
        model.load_state_dict(best_state["model"])
    final = {
        "train": evaluate(model, train_loader, device, args.ece_bins),
        "val": evaluate(model, val_loader, device, args.ece_bins),
        "test": evaluate(model, test_loader, device, args.ece_bins),
        "sam_score": {
            "train": evaluate_sam_score(train_records, args.ece_bins),
            "val": evaluate_sam_score(val_records, args.ece_bins),
            "test": evaluate_sam_score(test_records, args.ece_bins),
        },
        "split": {
            "split_by": args.split_by,
            "train": split_summary(train_records),
            "val": split_summary(val_records),
            "test": split_summary(test_records),
        },
    }

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, "uncertainty_head.pt")
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    torch.save({**best_state, "final_metrics": final}, ckpt_path)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)
    return {"checkpoint": ckpt_path, "metrics": metrics_path, "final": final}


def main() -> None:
    parser = argparse.ArgumentParser("Train a lightweight expected-IoU head on SANSA traces.")
    parser.add_argument("--cache_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="output/uncertainty_head")
    parser.add_argument("--feature_set", type=str, default="tokens", choices=sorted(FEATURE_KEYS))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--smooth_l1_beta", type=float, default=0.05)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--split_by", type=str, default="random", choices=["random", "class_id"])
    parser.add_argument("--ece_bins", type=int, default=10)
    parser.add_argument("--max_records", type=int, default=None)
    parser.add_argument("--eval_sam_score_only", action="store_true", default=False)
    args = parser.parse_args()

    result = train(args)
    print(json.dumps(result["final"], indent=2))
    if result["checkpoint"] is not None:
        print(f"Saved checkpoint to {result['checkpoint']}")
    print(f"Saved metrics to {result['metrics']}")


if __name__ == "__main__":
    main()
