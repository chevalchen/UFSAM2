import argparse
import sys
from typing import Any
from os.path import join
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import opts
from models.sansa.sansa import build_sansa
from datasets import build_dataset
from train_uncertainty_head import IoUHead, make_feature
from util.commons import make_deterministic, setup_logging, resume_from_checkpoint
import util.misc as utils
from util.promptable_utils import build_prompt_dict
from util.metrics import AverageMeter, Evaluator


def main(args: argparse.Namespace) -> float:
    setup_logging(args.output_dir, console="info", rank=0)
    make_deterministic(args.seed)
    print(args)

    if args.hflip_tta and args.uq_hflip_tta:
        raise ValueError("Use either --hflip_tta or --uq_hflip_tta, not both.")
    if args.uq_hflip_tta and not args.uq_head_ckpt:
        raise ValueError("--uq_hflip_tta requires --uq_head_ckpt.")

    model = build_sansa(
        args.sam2_version,
        args.adaptformer_stages,
        args.channel_factor,
        args.device,
        hflip_tta=args.hflip_tta,
    )
    device = torch.device(args.device)
    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if args.resume:
        resume_from_checkpoint(args.resume, model)

    print(f"number of params: {n_parameters}")
    print('Start inference')

    mIoU = eval_fss(model, args)
    return mIoU


def _squeeze_trace_tensor(trace: dict[str, Any], key: str) -> torch.Tensor | None:
    value = trace.get(key)
    if value is None:
        return None
    return value.squeeze(0).cpu()


def _aggregate_trace_tensor(traces: list[dict[str, Any]], key: str) -> torch.Tensor | None:
    values = [_squeeze_trace_tensor(trace, key) for trace in traces]
    values = [value for value in values if value is not None]
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return torch.stack(values).mean(dim=0)


def _merge_support_trace(query_trace: dict[str, Any], support_traces: list[dict[str, Any]]) -> dict[str, Any]:
    trace = dict(query_trace)
    for key in (
        "support_iou_token",
        "support_mask_token",
        "support_mask_tokens",
        "support_obj_ptr",
        "support_memory_summary",
    ):
        trace[key] = _aggregate_trace_tensor(support_traces, key)
    return trace


def load_uncertainty_head(ckpt_path: str, device: torch.device) -> tuple[nn.Module, tuple[str, ...], torch.Tensor, torch.Tensor]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    feature_keys = tuple(ckpt["feature_keys"])
    mean = ckpt["mean"].float()
    std = ckpt["std"].float().clamp_min(1e-6)
    head_args = ckpt.get("args", {})
    hidden_dim = int(head_args.get("hidden_dim", ckpt["model"]["net.0.weight"].shape[0]))
    dropout = float(head_args.get("dropout", 0.0))
    head = IoUHead(mean.numel(), hidden_dim, dropout).to(device)
    head.load_state_dict(ckpt["model"])
    head.eval()
    return head, feature_keys, mean.to(device), std.to(device)


@torch.no_grad()
def predict_expected_iou(
    head: nn.Module,
    feature_keys: tuple[str, ...],
    mean: torch.Tensor,
    std: torch.Tensor,
    trace: dict[str, Any],
    device: torch.device,
) -> float:
    feature = make_feature(trace, feature_keys).to(device)
    feature = (feature - mean) / std
    return head(feature.unsqueeze(0)).item()


def eval_fss(model: torch.nn.Module, args: argparse.Namespace) -> float:
    """
    Evaluate SANSA on the few-shot segmentation benchmark.
    Computes and prints mIoU across the validation set.
    """
    # load data
    validation_ds = 'coco' if args.dataset_file == 'multi' else args.dataset_file 
    print(f'Evaluating {validation_ds} - fold: {args.fold}')
    ds = build_dataset(validation_ds, image_set='val', args=args)
    dataloader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers)
    
    model.eval()
    average_meter = AverageMeter(args.dataset_file, ds.class_ids, ds.nclass)
    uq_state = None
    gated_hflip_count = 0
    if args.uq_hflip_tta:
        head_device = torch.device(args.uq_head_device)
        uq_state = load_uncertainty_head(args.uq_head_ckpt, head_device)

    max_episodes = len(dataloader) if args.max_eval_episodes is None else min(args.max_eval_episodes, len(dataloader))
    pbar = tqdm(dataloader, total=max_episodes, ncols=80, desc='runn avg.', disable=(utils.get_rank() != 0), file=sys.stderr, dynamic_ncols=True)
    for idx, batch in enumerate(pbar):
        if idx >= max_episodes:
            break
        query_img, query_mask = batch['query_img'], batch['query_mask']
        support_imgs, support_masks = batch['support_imgs'], batch['support_masks']

        imgs = torch.cat([support_imgs[0], query_img]).unsqueeze(0) # b t c h w
        img_h, img_w = imgs.shape[-2:]

        imgs = imgs.to(args.device)
        prompt_dict = build_prompt_dict(support_masks, args.prompt, n_shots=args.shots, train_mode=False, device=model.device)

        with torch.no_grad():
            if uq_state is None:
                outputs = model(imgs, prompt_dict)
            else:
                model.hflip_tta = False
                outputs = model(imgs, prompt_dict, return_traces=True)
                trace = _merge_support_trace(outputs["traces"][-1], outputs.get("support_traces", []))
                head, feature_keys, mean_vec, std_vec = uq_state
                expected_iou = predict_expected_iou(
                    head,
                    feature_keys,
                    mean_vec,
                    std_vec,
                    trace,
                    torch.device(args.uq_head_device),
                )
                if expected_iou < args.uq_gate_threshold:
                    gated_hflip_count += 1
                    model.hflip_tta = True
                    outputs = model(imgs, prompt_dict)
                model.hflip_tta = False

        pred_masks = outputs["pred_masks"].unsqueeze(0)  # [1, T, h, w]
        pred_masks = F.interpolate(pred_masks, size=(img_h, img_w), mode='bilinear', align_corners=False) 
        pred_masks = (pred_masks.sigmoid() > args.threshold)[0].cpu()

        area_inter, area_union = Evaluator.classify_prediction(pred_masks[-1:].float(), batch, device=imgs.device)
        average_meter.update(area_inter, area_union, batch['class_id'].cuda())

        if (idx + 1) % 50 == 0:
            miou, _, _ = average_meter.compute_iou()
            pbar.set_description(f"Runn. Avg mIoU = {miou:.1f}")

        if args.visualize:
            from util.visualization import visualize_episode
            visualize_episode(
                support_imgs=[support_imgs[0, i].cpu() for i in range(args.shots)],
                query_img=query_img[0].cpu(),
                query_gt=(query_mask[0].numpy() > 0),
                query_pred=pred_masks[-1].numpy(),
                prompt_dict=prompt_dict,
                out_dir=args.output_dir,
                idx=idx,
                src_size=model.sam.image_size,
                iou=area_inter/area_union,
            )
    average_meter.write_result(args.dataset_file)
    miou, fb_iou, _ = average_meter.compute_iou()
    print('Fold %d mIoU: %5.2f \t FB-IoU: %5.2f' % (args.fold, miou, fb_iou.item()))
    if args.uq_hflip_tta:
        print(f'UQ-gated hflip triggered on {gated_hflip_count}/{max_episodes} episodes at threshold {args.uq_gate_threshold:.3f}')
    print('==================== Finished Testing ====================')

    return miou


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SANSA evaluation script', parents=[opts.get_args_parser()])
    args = parser.parse_args()
    args.output_dir = join(args.output_dir, args.name_exp)
    main(args)
