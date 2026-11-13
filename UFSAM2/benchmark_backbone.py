"""Synthetic SANSA adapter-training cost benchmark; NOT an accuracy evaluation.

Run each model in a fresh process, on the same idle GPU with identical arguments.
Uses original SAM2 weights and newly initialized adapters for both model sizes.
"""
import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['small', 'large'], required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--shots', type=int, default=1)
    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--precision', choices=['fp32', 'bf16'], default='fp32')
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--steps', type=int, default=30)
    args = parser.parse_args()
    if min(args.batch_size, args.shots, args.size, args.warmup, args.steps) < 1:
        parser.error('All counts and sizes must be positive')
    output = Path(args.output)
    if output.exists():
        parser.error('Output already exists; choose a new run filename')

    import torch
    from models.sansa.sansa import build_sansa
    from util.losses import loss_masks
    from util.promptable_utils import build_prompt_dict

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required')
    if args.precision == 'bf16' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 is not supported on this GPU')
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = build_sansa(sam2_version=args.model, sam2_checkpoint=args.checkpoint,
                        adaptformer_stages=[2, 3], channel_factor=0.3).cuda().train()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-4)
    # Identical resident input for each backbone; excludes data loading and transfer.
    torch.manual_seed(123)
    x = torch.randn(args.batch_size, args.shots + 1, 3, args.size, args.size, device='cuda')
    masks = torch.zeros(args.batch_size, args.shots + 1, args.size, args.size, device='cuda')
    masks[..., args.size//4:3*args.size//4, args.size//4:3*args.size//4] = 1
    times, values = [], []
    for step in range(args.warmup + args.steps):
        optimizer.zero_grad(set_to_none=True)
        if step == args.warmup:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        prompts = build_prompt_dict(masks, 'mask', args.shots, True, model.device)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=args.precision == 'bf16'):
            pred = model(x, prompts)['pred_masks']
            loss = sum(loss_masks(pred, masks, num_frames=args.shots).values())
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if not torch.isfinite(loss).item() or not torch.isfinite(norm).item():
            raise RuntimeError('Nonfinite loss or gradient')
        if not any(p.grad is not None and p.grad.abs().max().item() > 0 for p in params):
            raise RuntimeError('No nonzero adapter gradient')
        if step >= args.warmup:
            times.append(elapsed)
            values.append(loss.item())
        del pred, loss, prompts, norm
    peak_alloc = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    digest = hashlib.sha256()
    with open(args.checkpoint, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    repo = Path(__file__).resolve().parent.parent
    def git(*cmd):
        return subprocess.check_output(['git', '-C', str(repo), *cmd], text=True).strip()
    result = dict(config=vars(args), kind='synthetic_adapter_training_cost_only',
                  gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                  cuda=torch.version.cuda, python=platform.python_version(),
                  git_sha=git('rev-parse', 'HEAD'), git_status=git('status', '--porcelain'),
                  checkpoint_sha256=digest.hexdigest(),
                  parameters=sum(p.numel() for p in model.parameters()),
                  trainable_parameters=sum(p.numel() for p in params),
                  internal_image_size=model.sam.image_size,
                  mean_step_seconds=statistics.mean(times), median_step_seconds=statistics.median(times),
                  episodes_per_second=args.batch_size/statistics.mean(times),
                  peak_allocated_gib=peak_alloc/2**30, peak_reserved_gib=peak_reserved/2**30,
                  step_seconds=times, losses=values)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
