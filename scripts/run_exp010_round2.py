"""EXP-010 ridge ablations and area-matched support granularity shuffle."""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

from run_exp010 import (
    FEATURES, SEED, build_sansa, collect, fit, legacy_ids, make_deterministic,
    native_metric, read_audit, sha,
)

PASCAL = tuple(f"pascal_f{fold}" for fold in range(4))
COHORTS = ("paco_dev", "paco_heldout", *PASCAL, "fss")
ABLATIONS = {
    "B": FEATURES[:6],
    "B_area": FEATURES[:6] + ["sqrt_area_gap"],
    "B_nesting": FEATURES[:6] + ["nesting_gap_zero_if_unknown", "nesting_valid"],
    "C": FEATURES,
}


def load(path):
    with np.load(path, allow_pickle=False) as archive:
        return {k: archive[k] for k in archive.files}


def pick(features, head, native):
    center, scale, weights, intercept = head
    score = ((features - center) / scale) @ weights + intercept
    return np.where(score[np.arange(len(native)), native] == score.max(1), native, score.argmax(1))


def score_cohorts(cache, train, old_heads, old_results):
    result = {}
    actual_heads = {}
    for mode, features in ABLATIONS.items():
        cols = [FEATURES.index(name) for name in features]
        head = fit({**train, "features": train["features"][..., cols]}, len(cols))
        if mode in ("B", "C"):
            saved = old_heads[mode]
            for value, key in zip(head, ("center", "scale", "weights", "intercept")):
                np.testing.assert_allclose(value, saved[key], rtol=1e-10, atol=1e-12)
        actual_heads[mode] = (cols, head)
    for cohort in COHORTS:
        ds = cache[cohort]
        result[cohort] = {}
        for mode, (cols, head) in actual_heads.items():
            metric = native_metric(ds, pick(ds["features"][..., cols], head, ds["native"]))
            if mode in ("B", "C"):
                for k, value in metric.items():
                    np.testing.assert_allclose(value, old_results[cohort][mode][k], rtol=0, atol=1e-10)
            result[cohort][mode] = metric
    result["pascal_mean"] = {
        mode: {metric: float(np.mean([result[fold][mode][metric] for fold in PASCAL]))
               for metric in ("miou", "fb_iou")} for mode in ABLATIONS
    }
    return result, actual_heads["C"][1]


def raw_replay(args, originals):
    torch.set_num_threads(4)
    make_deterministic(SEED)
    model = build_sansa("small", [2, 3], .8, device="cuda",
                        sam2_checkpoint=str(args.base)).cuda().eval().requires_grad_(False)
    saved = torch.load(args.adapter, map_location="cpu", weights_only=False)
    assert saved["epoch"] == 4
    expected = {name for name, _ in model.named_parameters() if "adapter" in name}
    assert expected == saved["model"].keys()
    load_status = model.load_state_dict(saved["model"], strict=False)
    assert not load_status.unexpected_keys
    model.audit_candidates = True
    backbone = model._forward_backbone

    def capture(*a, **kw):
        model.exp010_backbone = backbone(*a, **kw)
        return model.exp010_backbone

    model._forward_backbone = capture
    audit = read_audit(args.audit)
    assert len(audit) == 2500
    dev_ids = {img for row in audit for img in legacy_ids(row)}
    replay = collect(model, args.data, "paco_part", "val", 1,
                     range(len(originals["native"])), SEED, args.output / "heldout_replay.npz",
                     forbidden=dev_ids, save_granularity=True)
    for field in ("query", "support", "labels", "class_ids", "native", "inter", "union", "pixels", "gt"):
        np.testing.assert_array_equal(replay[field], originals[field], err_msg=field)
    np.testing.assert_allclose(replay["features"], originals["features"], rtol=0, atol=1e-7)
    return replay["granularity"]


def with_granularity(features, descriptor, donor):
    updated = features.copy()
    updated[..., 6] = np.abs(features[..., 1] - descriptor[donor, 0, None])
    gap = np.abs(descriptor[:, 2:] - descriptor[donor, 1, None])
    updated[..., 7] = np.nan_to_num(gap)
    updated[..., 8] = np.isfinite(gap)
    return updated


def area_shuffle(ds, descriptor, head, output):
    count = len(descriptor)
    assert descriptor.shape == (count, 5)
    original = ds["features"]
    np.testing.assert_allclose(with_granularity(original, descriptor, np.arange(count)),
                               original, rtol=0, atol=1e-7)
    # Equal-frequency groups with stable ordering for tied areas.
    bins = np.array_split(np.argsort(descriptor[:, 0], kind="stable"), 10)
    assignments, scores, area_errors = [], [], []
    native = ds["native"]
    actual = pick(original, head, native)
    actual_metric = native_metric(ds, actual)
    for seed in range(30):
        rng = np.random.default_rng(seed)
        donor = np.full(count, -1, dtype=int)
        for group in bins:
            support = ds["support"][group]
            same = support[:, None] == support[None, :]
            cost = rng.random(same.shape)
            cost[same] = np.inf
            row, col = linear_sum_assignment(cost)
            donor[group[row]] = group[col]
            assert np.array_equal(np.sort(donor[group]), np.sort(group))
        assert sorted(donor) == list(range(count))
        assert np.all(ds["support"][donor] != ds["support"])
        metric = native_metric(ds, pick(with_granularity(original, descriptor, donor), head, native))
        scores.append(dict(seed=seed, **metric))
        area_errors.append(np.abs(descriptor[:, 0]**2 - descriptor[donor, 0]**2))
        assignments.append(donor)
    np.save(output / "donor_indices.npy", np.asarray(assignments), allow_pickle=False)
    miou = np.asarray([x["miou"] for x in scores])
    area_errors = np.concatenate(area_errors)
    return dict(real=actual_metric, shuffled=scores,
                miou_mean=float(miou.mean()), miou_std=float(miou.std(ddof=1)),
                miou_min=float(miou.min()), miou_max=float(miou.max()),
                real_minus_shuffle_mean=float(actual_metric["miou"] - miou.mean()),
                real_above_seed_count=int((actual_metric["miou"] > miou).sum()),
                absolute_support_area_gap={k: float(v) for k, v in
                                           zip(("mean","p95","max"),
                                               (area_errors.mean(), np.quantile(area_errors,.95), area_errors.max()))})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pilot", "output", "base", "adapter", "audit"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("/data6/chensq/datasets"))
    args = parser.parse_args()
    assert all(path.is_absolute() for path in vars(args).values())
    args.output.mkdir(parents=True, exist_ok=False)
    inputs = [args.pilot / f"{name}.npz" for name in (*COHORTS, "paco_train_f0", "paco_train_f1")]
    source_manifest = json.loads((args.pilot / "manifest.json").read_text())
    assert source_manifest["features"] == FEATURES and source_manifest["ridge_lambda"] == .01
    old_results = json.loads((args.pilot / "summary.json").read_text())
    old_heads = json.loads((args.pilot / "heads.json").read_text())
    for path in [args.base, args.adapter, args.audit, *sorted((args.data / "PACO-Part/paco").glob("paco_part_*.pkl"))]:
        assert sha(path) == source_manifest["input_sha256"][str(path)]
    cache = {p.stem: load(p) for p in inputs}
    train = {k: np.concatenate([cache[f"paco_train_f{fold}"][k] for fold in (0, 1)])
             for k in ("features", "inter", "union")}
    assert len(train["features"]) == 3000
    result, head = score_cohorts(cache, train, old_heads, old_results)
    (args.output / "ablations.json").write_text(json.dumps(result, indent=2) + "\n")
    manifest = dict(sha=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                    arguments={k: str(v) for k, v in vars(args).items()},
                    torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                    tf32_matmul=torch.backends.cuda.matmul.allow_tf32, tf32_cudnn=torch.backends.cudnn.allow_tf32,
                    pilot_sha=source_manifest["sha"], pilot_cache_sha256={str(p): sha(p) for p in inputs},
                    features=ABLATIONS, area_quantile_bins=10, shuffle_seeds=list(range(30)))
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Cached ablations completed; B and C reproduced exactly on all cohorts.", flush=True)
    heldout = cache["paco_heldout"]
    assert len(heldout["native"]) == 1000 and len(heldout["class_ids"]) == 60
    descriptor = raw_replay(args, heldout)
    result["shuffle_heldout"] = area_shuffle(heldout, descriptor, head, args.output)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print("Verified replay and finished 30 area-matched shuffles.", flush=True)


if __name__ == "__main__":
    main()
