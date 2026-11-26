"""Full paired EXP-010 evaluation. Run prepare once, then one worker per shot."""
import argparse
import json
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

from run_exp010 import (FEATURES, build_dataset, build_prompt_dict, build_sansa,
                        episode_features, fit, make_deterministic, np, rescale_prompt, torch, F)
from torch.utils.data import DataLoader
from util.metrics import AverageMeter, Evaluator


def prepare(root, out):
    pilot = root / "exp010_artifacts/pilot_v3"
    parts = [np.load(pilot / f"paco_train_f{i}.npz") for i in (0, 1)]
    train = {k: np.concatenate([p[k] for p in parts]) for k in ("inter", "union", "features")}
    head = fit(train, 7)
    manifest = dict(features=FEATURES[:7], head=[np.asarray(v).tolist() for v in head],
                    sha=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                    pilot=str(pilot), seed=0, ridge_lambda=.01, torch=torch.__version__,
                    base="/data6/chensq/SANSA/pretrain/sam2_hiera_small.pt",
                    adapter=str(root / "exp008_artifacts/train_v1/checkpoint0004.pth"))
    with (out / "manifest.json").open("x") as f:
        json.dump(manifest, f, indent=2)


@torch.no_grad()
def evaluate(model, dataset, fold, shots, head, out):
    make_deterministic(0)
    args = SimpleNamespace(data_root="/data6/chensq/datasets", fold=fold, shots=shots)
    ds = build_dataset(dataset, "val", args)
    meters = [AverageMeter(dataset, ds.class_ids, ds.nclass) for _ in range(2)]
    start = time.monotonic()
    center, scale, weights, intercept = head
    switches = 0
    tag = f"{dataset}_s{shots}_f{fold}"
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    for i, batch in enumerate(loader, 1):
        imgs = torch.cat([batch["support_imgs"][0], batch["query_img"]])[None].cuda()
        prompts = build_prompt_dict(batch["support_masks"], "mask", shots, False, model.device)
        result = model(imgs, prompts)
        logits, quality = result["candidate_logits"][0], result["candidate_quality"][0]
        scores = []
        for j in range(shots):
            mask = prompts[0][j]["prompt"]
            resized = rescale_prompt(mask, "mask", imgs.shape[-2:], model.sam.image_size)
            ref = (resized > 0).float().mean().sqrt().reshape(1) if resized.any() else None
            features = episode_features(model.exp010_backbone, logits, quality, mask[0], ref,
                                        support_index=j, query_index=shots, area_only=True)
            scores.append(((features.double() - center) / scale) @ weights + intercept)
        score = torch.stack(scores).mean(0)
        native = result["selected_candidate_index"].item()
        best = score.argmax().item()
        selected = native if score[native] == score[best] else best
        switches += selected != native
        masks = F.interpolate(logits[None], size=imgs.shape[-2:], mode="bilinear",
                              align_corners=False)[0].sigmoid() > .5
        for meter, choice in zip(meters, (native, selected)):
            inter, union = Evaluator.classify_prediction(masks[choice:choice+1].float(), batch, imgs.device)
            meter.update(inter, union, batch["class_id"].cuda())
        if i == 1 or i % 250 == 0:
            print(tag, i, "/", len(ds), flush=True)
    metrics = {}
    for key, meter in zip(("native", "area"), meters):
        miou, fb, _ = meter.compute_iou()
        metrics[key] = dict(miou=miou, fb_iou=fb.item())
    row = dict(dataset=dataset, fold=fold, shots=shots, episodes=len(ds), switches=switches,
               classes=len(ds.class_ids), seconds=time.monotonic()-start, **metrics)
    tmp = out / f"{tag}.tmp"
    tmp.write_text(json.dumps(row, indent=2))
    tmp.rename(out / f"{tag}.json")
    print(row, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "1", "5"])
    parser.add_argument("output", type=Path)
    parser.add_argument("--root", type=Path, default=Path("/data6/chensq/UFSAM2"))
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(exist_ok=True, parents=True)
    if args.mode == "prepare":
        prepare(args.root, out)
        return
    cfg = json.loads((out / "manifest.json").read_text())
    head = [torch.as_tensor(x, dtype=torch.float64, device="cuda") for x in cfg["head"]]
    torch.set_num_threads(4)
    make_deterministic(0)
    model = build_sansa("small", [2, 3], .8, device="cuda", sam2_checkpoint=cfg["base"]).cuda().eval()
    model.load_state_dict(torch.load(cfg["adapter"], map_location="cpu", weights_only=False)["model"], strict=False)
    model.audit_candidates = True
    backbone = model._forward_backbone

    def capture(*args, **kwargs):
        model.exp010_backbone = backbone(*args, **kwargs)
        return model.exp010_backbone

    model._forward_backbone = capture
    shots = int(args.mode)
    for dataset, folds in [("pascal_part", range(4)), ("fss", [-1]), ("paco_part", range(4))]:
        for fold in folds:
            if not (out / f"{dataset}_s{shots}_f{fold}.json").exists():
                evaluate(model, dataset, fold, shots, head, out)


if __name__ == "__main__":
    main()
