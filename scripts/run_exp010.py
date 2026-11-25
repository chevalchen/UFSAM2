"""Cache frozen SANSA candidates and compare three nested ridge scoring heads."""
import argparse
import ast
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT / "UFSAM2")
sys.path.insert(0, str(ROOT / "UFSAM2"))
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
from datasets import build_dataset
from models.sansa.granularity import ReferenceGranularity, nesting_level
from models.sansa.sansa import build_sansa
from util.commons import make_deterministic
from util.promptable_utils import build_prompt_dict

FEATURES = [
    "quality", "sqrt_area", "mean_sibling_iou", "max_sibling_iou",
    "support_foreground_cosine", "support_near_background_cosine",
    "sqrt_area_gap", "nesting_gap_zero_if_unknown", "nesting_valid",
]
MODES = {"A": 4, "B": 6, "C": 9}
SEED = 0


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else hash_stream(stream)


def hash_stream(stream):
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def read_audit(path):
    with path.open() as f:
        return [json.loads(line.split("candidate_audit ", 1)[1])
                for line in f if "candidate_audit " in line]


def image_ids(sample):
    return [Path(str(x)).stem for x in [sample["query_name"], *sample["support_names"]]]


def legacy_ids(row):
    return image_ids(dict(query_name=ast.literal_eval(row["query"])[0],
                          support_names=[entry[0] for entry in ast.literal_eval(row["supports"])]))


def restrict_paco(ds, forbidden, unique):
    before = len(ds.class_ids_ori)
    for cid in ds.class_ids_ori:
        entries = {}
        pool = []
        for entry in ds.cid2img[cid]:
            name = next(iter(entry.values()))
            key = Path(name).stem
            if key not in forbidden:
                entries.setdefault(key, entry)
                pool.append(entry)
        ds.cid2img[cid] = list(entries.values()) if unique else pool
    ds.class_ids_ori = [cid for cid in ds.class_ids_ori if len({
        Path(next(iter(entry.values()))).stem for entry in ds.cid2img[cid]}) >= 2]
    ds.class_ids_c = {cid: i for i, cid in enumerate(ds.class_ids_ori)}
    ds.class_ids = list(range(len(ds.class_ids_ori)))
    assert ds.class_ids_ori
    print("PACO pool classes", before, "->", len(ds.class_ids_ori), flush=True)


class FrozenSelector(ReferenceGranularity):
    """Reuse support probing, while preserving the original query choice."""
    def __init__(self):
        super().__init__("nesting")
        self.reference = None

    def forward(self, logits, quality, references):
        if references:
            self.reference = references[0]
        return quality.argmax(dim=-1)


def weighted_pool(features, regions):
    weights = F.interpolate(regions[:, None].float(), size=features.shape[-2:], mode="area")[:, 0]
    weights = weights.flatten(1)
    pooled = weights @ features.flatten(1).T / weights.sum(1, keepdim=True).clamp_min(1e-6)
    return F.normalize(pooled, dim=1)


def episode_features(backbone, logits, quality, support, reference):
    masks = logits > 0
    flat = masks.flatten(1).float()
    inter = flat @ flat.T
    area = flat.sum(1)
    pair_iou = inter / (area[:, None] + area[None, :] - inter).clamp_min(1)
    pair_iou.fill_diagonal_(0)
    occupancy = flat.mean(1).sqrt()
    supports = F.interpolate(support[:, None].float(), size=logits.shape[-2:], mode="nearest")[:, 0] > 0
    ring = (F.max_pool2d(supports[None].float(), 33, stride=1, padding=16)[0] > 0) & ~supports
    support_desc = weighted_pool(backbone.get_current_feats_x16(0)[0], torch.cat([supports, ring]))
    query_desc = weighted_pool(backbone.get_current_feats_x16(1)[0], masks)
    similarities = query_desc @ support_desc.T
    if reference is None:
        reference = torch.stack([supports.float().mean().sqrt(), occupancy.new_tensor(float("nan"))])
    gap = (nesting_level(masks, masks) - reference[1]).abs()
    return torch.column_stack([quality, occupancy, pair_iou.sum(1) / 2, pair_iou.max(1).values,
                               similarities, (occupancy - reference[0]).abs(),
                               gap.nan_to_num(), gap.isfinite().float()])


def diagnostic_panel(path, batch, candidates, native, best, ious):
    mean = torch.tensor([.485,.456,.406])[:,None,None]
    std = torch.tensor([.229,.224,.225])[:,None,None]
    def paint(image, mask):
        rgb = ((image.cpu() * std + mean).clamp(0,1).permute(1,2,0).numpy()*255).astype("uint8")
        rgb[mask] = (.5*rgb[mask] + .5*np.array([255, 70, 30])).astype("uint8")
        return Image.fromarray(rgb).resize((224,224))
    canvas = Image.new("RGB", (1120,248), "white")
    draw = ImageDraw.Draw(canvas)
    query = batch["query_img"]
    entries = [(batch["support_imgs"][0], batch["support_masks"][0].numpy()>0, "Support GT"),
               (query, batch["query_mask"].numpy()>0, "Query GT (diagnostic)")]
    for k,mask in enumerate(candidates):
        tag = (" native" if k==native else "") + (" best" if k==best else "")
        entries.append((query, mask.cpu().numpy(), f"C{k}{tag} IoU={ious[k]:.2f}"))
    for k,(img,mask,title) in enumerate(entries):
        canvas.paste(paint(img,mask),(k*224,24))
        draw.text((k*224+4,5), title, fill="black")
    canvas.save(path)


def collect(model, selector, data, ds_name, split, fold, indices, seed, path, forbidden=None,
            old_audit=None, diagnostics=()):
    from types import SimpleNamespace
    make_deterministic(seed)
    dataset = build_dataset(ds_name, split, SimpleNamespace(data_root=str(data), fold=fold, shots=1))
    if forbidden is not None:
        restrict_paco(dataset, forbidden, unique=True)
    make_deterministic(seed)
    cache = {key: [] for key in ("features","inter","union","gt","pixels","native","labels","query","support")}
    with torch.no_grad():
        for row, idx in enumerate(indices):
            batch = dataset[idx]
            query, support = image_ids(batch)
            if forbidden is not None:
                assert query not in forbidden and support not in forbidden
            samples = torch.cat([batch["support_imgs"],batch["query_img"][None]])[None].cuda()
            prompt = build_prompt_dict(batch["support_masks"][None],"mask",1,False,model.device)
            selector.reference = None
            result = model(samples,prompt)
            logits = result["candidate_logits"][0]
            quality = result["candidate_quality"][0]
            assert len(logits)==3
            target = batch["query_mask"].cuda()>0
            candidates = F.interpolate(logits[None], size=target.shape, mode="bilinear",align_corners=False)[0]>0
            inter = (candidates & target).sum((-2,-1)).cpu().numpy()
            union = (candidates | target).sum((-2,-1)).cpu().numpy()
            native = quality.argmax().item()
            assert result["selected_candidate_index"].item()==native
            assert torch.equal(result["pred_masks"][-1]>0,candidates[native])
            label = batch["class_id"].item()
            if old_audit is not None:
                old = old_audit[idx]
                assert legacy_ids(old)==[query,support] and label==old["class"]
                assert old["native_index"]==native and list(inter)==old["intersections"] and list(union)==old["unions"]
                if idx in diagnostics:
                    iou=inter/np.maximum(union,1)
                    diagnostic_panel(path.parent/f"diagnostic-{idx:04d}.png",batch,candidates,native,int(iou.argmax()),iou)
            values = (episode_features(model.exp010_backbone,logits,quality,batch["support_masks"].cuda(),
                                       selector.reference).cpu().numpy(), inter,union,target.sum().item(),target.numel(),
                      native,label,query,support)
            for col,value in zip(cache,values):
                cache[col].append(value)
            if row%100==0 or row+1==len(indices):
                print(path.stem, row+1, "/", len(indices), flush=True)
    arrays={k:np.asarray(v) for k,v in cache.items()}
    arrays["class_ids"]=np.asarray(dataset.class_ids)
    assert np.isfinite(arrays["features"]).all()
    np.savez_compressed(path,**arrays)
    print("SAVED",path,flush=True)
    return arrays


def native_metric(data, selected):
    ii = data["inter"][np.arange(len(selected)),selected]
    uu = data["union"][np.arange(len(selected)),selected]
    classes = data["class_ids"]
    inter=np.bincount(data["labels"],weights=ii,minlength=int(classes.max())+1)[classes]
    union=np.bincount(data["labels"],weights=uu,minlength=int(classes.max())+1)[classes]
    background = (data["pixels"]-uu).sum()/max(1,(data["pixels"]-ii).sum())
    fg = ii.sum()/max(1,uu.sum())
    return dict(miou=float(100*np.mean(inter/np.maximum(union,1))), fb_iou=float(50*(fg+background)))


def fit(train,width):
    X=train["features"][:,:,:width].reshape(-1,width).astype("float64")
    y=(train["inter"]/np.maximum(train["union"],1)).reshape(-1)
    center=X.mean(0)
    scale=X.std(0).clip(1e-6)
    X=(X-center)/scale
    weights=np.linalg.solve(X.T@X + len(X)*.01*np.eye(width), X.T@(y-y.mean()))
    return center,scale,weights,float(y.mean())


def evaluate(train, data):
    native=data["native"]
    metrics={"native":native_metric(data,native)}
    for mode,width in MODES.items():
        center,scale,weights,intercept=fit(train,width)
        scores=(data["features"][:,:,:width]-center)/scale@weights + intercept
        picked=scores.argmax(1)
        pick_ties=scores[np.arange(len(native)),native]==scores.max(1)
        picked[pick_ties]=native[pick_ties]
        switched=picked!=native
        row=np.arange(len(native))
        delta=(data["inter"][row,picked]*np.maximum(data["union"][row,native],1)
                 - data["inter"][row,native]*np.maximum(data["union"][row,picked],1))
        metrics[mode]=dict(**native_metric(data,picked),switches=int(switched.sum()),
                           improved=int((switched & (delta>0)).sum()),worsened=int((switched & (delta<0)).sum()),
                           unchanged=int((switched & (delta==0)).sum()))
    return metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    parser.add_argument("--data",type=Path,default=Path("/data6/chensq/datasets"))
    parser.add_argument("--base",type=Path,required=True)
    parser.add_argument("--adapter",type=Path,required=True)
    parser.add_argument("--audit",type=Path,required=True)
    parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    make_deterministic(SEED)
    assert subprocess.check_output(["git","status","--porcelain"],text=True).strip()==""
    model=build_sansa("small",[2,3],.8,device="cuda",sam2_checkpoint=str(args.base)).cuda().eval().requires_grad_(False)
    saved=torch.load(args.adapter,map_location="cpu",weights_only=False)
    assert saved["epoch"]==4
    expected={name for name,_ in model.named_parameters() if "adapter" in name}
    assert expected==saved["model"].keys()
    model.load_state_dict(saved["model"],strict=False)
    selector=FrozenSelector()
    model.granularity=selector
    model.audit_candidates=True
    forward_backbone=model._forward_backbone
    def capture(*a,**kw):
        model.exp010_backbone=forward_backbone(*a,**kw)
        return model.exp010_backbone
    model._forward_backbone=capture
    audit=read_audit(args.audit)
    assert len(audit)==2500
    dev_ids={x for r in audit for x in legacy_ids(r)}
    val_path=args.data/"PACO-Part/paco/paco_part_val.pkl"
    train_path=val_path.with_name("paco_part_train.pkl")
    with val_path.open("rb") as f:
        validation=pickle.load(f)
    val_ids={Path(str(next(iter(item.values())))).stem for pool in validation["cid2img"].values() for item in pool}
    del validation
    assert dev_ids<=val_ids
    inputs=[args.base,args.adapter,args.audit,train_path,val_path,
            args.data/"FSS-1000/splits/test.txt",
            args.data/"Pascal-Part/VOCdevkit/VOC2010/all_obj_part_to_image.json"]
    manifest=dict(sha=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
                  arguments=sys.argv,torch=torch.__version__,cuda=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(),tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                  tf32_cudnn=torch.backends.cudnn.allow_tf32,
                  features=FEATURES,seed=SEED,train_folds=[0,1],ridge_lambda=.01,
                  input_sha256={str(p):sha(p) for p in inputs})
    (args.output/"manifest.json").write_text(json.dumps(manifest,indent=2))
    def cache(name, ds, split, fold, indices, seed=0, **kwargs):
        if args.smoke:
            indices=list(indices)[:4]
        return collect(model,selector,args.data,ds,split,fold,indices,seed,args.output/f"{name}.npz",**kwargs)
    def gap(r):
        iou=np.asarray(r["intersections"])/np.maximum(r["unions"],1)
        return iou.max()-iou[r["native_index"]]
    top=sorted((r["episode"] for r in sorted(audit,key=gap,reverse=True)[:6]))
    if args.smoke:
        top=list(range(4))
    train_parts=[]
    dev=cache("paco_dev","paco_part","val",0,range(2500),old_audit=audit,diagnostics=top)
    for fold in (0,1):
        train_parts.append(cache(f"paco_train_f{fold}","paco_part","train",fold,range(1500),
                                 seed=101+fold,forbidden=val_ids))
    train={k:np.concatenate([p[k] for p in train_parts]) for k in ("features","inter","union")}
    heldout=cache("paco_heldout","paco_part","val",1,range(1000),seed=0,forbidden=dev_ids)
    def ids(arr): return set(arr["query"]) | set(arr["support"])
    train_ids=set().union(*(ids(p) for p in train_parts))
    assert not train_ids.intersection(ids(dev) | ids(heldout))
    assert not ids(dev).intersection(ids(heldout)) and not dev_ids.intersection(ids(heldout))
    output={"paco_dev":evaluate(train,dev),"paco_heldout":evaluate(train,heldout)}
    fit_params={}
    for mode,width in MODES.items():
        center,scale,weights,intercept=fit(train,width)
        fit_params[mode]=dict(features=FEATURES[:width],center=center.tolist(),scale=scale.tolist(),
                              weights=weights.tolist(),intercept=intercept)
    (args.output/"heads.json").write_text(json.dumps(fit_params,indent=2))
    for fold in range(4):
        part=cache(f"pascal_f{fold}","pascal_part","val",fold,range(250))
        output[f"pascal_f{fold}"]=evaluate(train,part)
    fss=cache("fss","fss","val",0,[10*c+i for c in range(240) for i in (0,5)])
    output["fss"]=evaluate(train,fss)
    (args.output/"summary.json").write_text(json.dumps(output,indent=2))
    print(json.dumps(output,indent=2),flush=True)


if __name__=="__main__":
    main()
