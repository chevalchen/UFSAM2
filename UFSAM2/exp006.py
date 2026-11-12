"""Matched SANSA gradient self-prompting development experiment; no model training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F
from models.sansa.gradient_prompting import GradientPrompting
from util.exp004_data import sha256, verified_file
from util.exp004_runtime import build_model, seed_all, synchronize
from util.exp005_data import frame_data
from util.exp005_metrics import boundary_counts, summarize
from util.exp004_metrics import pixel_counts, episode_iou
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def state_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_episode(e, shots, device):
    transform = transforms.Compose([transforms.Resize((640, 640)), transforms.ToTensor(),
                                    transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    images, masks = [], []
    for frame in e['supports'][:shots] + [e['query']]:
        im, mask = frame_data(frame)
        images.append(transform(im))
        masks.append(F.interpolate(mask[None, None], (640, 640), mode='nearest')[0, 0])
    # Target returned separately; prepare/generate accept only support masks.
    return torch.stack(images)[None].to(device), torch.stack(masks[:-1]).to(device), masks[-1].to(device)


def resized(out, size):
    return F.interpolate(out.low_res_masks.detach(), size, mode='bilinear', align_corners=False)[0, 0]


def timed(device, function):
    synchronize(device); start = time.perf_counter()
    result = function()
    synchronize(device)
    return result, time.perf_counter() - start


def point_record(module, points, target, previous_logits):
    if points is None:
        return dict(coords=[], labels=[], hits=[], error_hits=[])
    xy = points['point_coords'][0].detach().cpu().tolist()
    labels = points['point_labels'][0].detach().cpu().tolist()
    cells = module.cells(points, target.shape, module.sansa.sam.image_size)
    hits, error_hits = [], []
    for (y, x), label in zip(cells, labels):
        gt = bool(target[y, x] == 1); pred = bool(previous_logits[y, x] > 0)
        hits.append(gt == bool(label)); error_hits.append(gt == bool(label) and gt != pred)
    return dict(coords=xy, labels=labels, hits=hits, error_hits=error_hits)


@torch.no_grad()
def gt_correction(module, out, target, previous):
    """ORACLE diagnostic only, outside the candidate generator."""
    from scipy.ndimage import distance_transform_edt, label
    z = resized(out, target.shape)
    gt, pred = (target == 1).cpu().numpy(), (z > 0).cpu().numpy()
    yy, xx = np.indices(gt.shape); allowed = np.ones(gt.shape, dtype=bool)
    radius = module.config['dedup_radius_lowres'] * gt.shape[0] / out.low_res_masks.shape[-2]
    for y, x in module.cells(previous, gt.shape, module.sansa.sam.image_size):
        allowed &= (yy-y)**2+(xx-x)**2 > radius**2
    chosen, labels, error_types = [], [], []
    for domain, fallback, lab in [(gt & ~pred, gt, 1), (~gt & pred, ~gt, 0)]:
        domain = domain & allowed
        error_types.append(bool(domain.any()))
        if not domain.any():
            domain = fallback & allowed
        if not domain.any():
            if lab == 1:
                return None, error_types
            continue
        components, _ = label(domain)
        areas = np.bincount(components.ravel()); areas[0] = 0
        domain = components == areas.argmax()
        distance = distance_transform_edt(np.pad(domain, 1))[1:-1, 1:-1]
        y, x = np.unravel_index(distance.argmax(), gt.shape)
        chosen.append((int(y), int(x))); labels.append(lab)
        allowed &= (yy-y)**2+(xx-x)**2 > radius**2
    return module.pack(chosen, labels, target), error_types


def generate(module, cache, seed):
    """Generate all deployable candidates before query GT diagnostics/metrics."""
    device = cache.feature.device
    saved = cache.feature.clone()
    timing = {}; outputs = {}; points = {}; stats = {}; maps = {}
    with torch.no_grad():
        outputs['B0'], timing['M0'] = timed(device, lambda: module.decode(cache))
        p0, timing['P0'] = timed(device, lambda: module.points(outputs['B0']))
    points['S1'] = p0
    if p0 is None:
        for row in ['S1', 'S2', 'G2', 'G2shuffle', 'H2']:
            outputs[row] = outputs['B0']; points[row] = None
        stats.update(initial_valid=False, gradient_valid=False)
    else:
        # The M1 graph is shared with the G2 backward, without a fourth candidate head call.
        temporary = cache.feature.detach().clone().requires_grad_(True)
        m1, timing['M1_graph'] = timed(device, lambda: module.decode(cache, p0, temporary))
        outputs['S1'] = module.detach(m1)
        (response, gradient_stats), timing['backward'] = timed(device, lambda: module.gradient(temporary, m1))
        stats.update(gradient_stats, initial_valid=True)
        del m1, temporary
        with torch.no_grad():
            for row in ['S2', 'G2', 'G2shuffle', 'H2']:
                def sample():
                    if row == 'H2':
                        return module.semantic_points(cache, outputs['S1'], p0)
                    return module.points(outputs['S1'], p0,
                        response=response if row != 'S2' else None,
                        shuffle_seed=seed if row == 'G2shuffle' else None)
                points[row], timing['sample_'+row] = timed(device, sample)
                if points[row] is None:
                    outputs[row] = outputs['S1']; timing[row] = 0.
                else:
                    outputs[row], timing[row] = timed(device, lambda: module.decode(cache, points[row]))
            if response is not None:
                pos, neg = module.domains(outputs['S1'])
                maps['gradient_response'] = response.cpu().numpy()
                maps['positive_domain'] = (pos >= module.config['positive_threshold']).cpu().numpy()
                maps['negative_domain'] = (neg >= 1e-6).cpu().numpy()
                pred = (outputs['S1'].low_res_masks > 0).float()
                boundary = (F.max_pool2d(pred,9,1,4) + F.max_pool2d(-pred,9,1,4))[0,0] > 0
                top = response >= torch.quantile(response,.9)
                stats['gradient_top10_boundary_fraction'] = float(boundary[top].float().mean())
                stats['boundary_image_fraction'] = float(boundary.float().mean())
                # Coverage is measured later using target, without changing these domains.
                stats['positive_domain_fraction'] = float((pos >= module.config['positive_threshold']).float().mean())
                stats['response_peak'] = list(divmod(int(response.flatten().argmax()), response.shape[1]))
    if not torch.equal(saved, cache.feature):
        raise RuntimeError('F0 modified by refinement')
    if any(p.requires_grad or p.grad is not None for p in module.sansa.parameters()):
        raise RuntimeError('Frozen model received gradients')
    stats['f0_unchanged'] = True
    return outputs, points, timing, stats, maps


def cluster_compare(a, b, iterations):
    keys = sorted({r['query_id'] for r in a}); rng = np.random.default_rng(6006)
    group = {k: [i for i, r in enumerate(a) if r['query_id'] == k] for k in keys}
    effects = np.array([episode_iou(x['counts'])-episode_iou(y['counts']) for x, y in zip(a, b)])
    values = []
    for _ in range(iterations):
        indices = [i for k in rng.choice(keys, len(keys), replace=True) for i in group[k]]
        x, y = [a[i] for i in indices], [b[i] for i in indices]
        # Each replicate recomputes per-class intersection/union, then equal class weights.
        from util.exp004_metrics import aggregate
        values.append([aggregate(x)['class_miou']-aggregate(y)['class_miou'], 100*effects[indices].mean()])
    ci = np.percentile(values, [2.5, 97.5], axis=0)
    return dict(class_delta_pp=summarize(a)['class_miou']-summarize(b)['class_miou'],
                episode_delta_pp=100*float(effects.mean()), worse_fraction=float((effects < -1e-8).mean()),
                better_fraction=float((effects > 1e-8).mean()), class_ci95_pp=ci[:, 0].tolist(),
                episode_ci95_pp=ci[:, 1].tolist(), query_clusters=len(keys))


def analyze(output, iterations):
    result = {}
    for path in sorted(output.glob('paired_*shot.jsonl')):
        records = [json.loads(line) for line in path.read_text().splitlines()]
        rows = {row: [r['rows'][row] for r in records] for row in records[0]['rows']}
        metrics = {row: summarize(values) for row, values in rows.items()}
        comparisons = {}
        for row, control in [('G2', 'S2'), ('G2', 'B0'), ('S1', 'B0'), ('S2', 'B0'), ('H2', 'G2'), ('G2', 'G2shuffle')]:
            comparisons[row+'-'+control] = cluster_compare(rows[row], rows[control], iterations)
        diagnostics = [r for r in records if 'GT2' in r.get('diagnostics', {})]
        gt = {}
        if diagnostics:
            gtrows = [r['diagnostics']['GT2'] for r in diagnostics]
            gt = dict(episodes=len(diagnostics), rows={row:summarize([r['rows'][row] for r in diagnostics]) for row in ['B0', 'S1', 'G2']},
                      GT2=summarize(gtrows), comparison=cluster_compare(gtrows, [r['rows']['S1'] for r in diagnostics], iterations))
        result[path.stem] = dict(episodes=len(records), rows=metrics, comparisons=comparisons, gt_diagnostic=gt,
                                mean_timing_seconds={key:float(np.mean([r['timing'].get(key, 0) for r in records])) for key in sorted({k for r in records for k in r['timing']})},
                                peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in records),
                                valid_initial_fraction=float(np.mean([r['stats']['initial_valid'] for r in records])),
                                valid_gradient_fraction=float(np.mean([r['stats']['gradient_valid'] for r in records])))
    write(output/'results.json', result)
    return result


def main():
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['pilot', 'run', 'analyze'])
    p.add_argument('--config', default='experiments/EXP-006/config_v1.json'); p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda:1'); p.add_argument('--limit', type=int)
    a = p.parse_args(); config_path = ROOT/a.config
    c = json.loads(config_path.read_text()); output = Path(a.output).resolve()
    if a.action == 'analyze':
        analyze(output, c['bootstrap_iterations']); return
    dirty = git('status', '--porcelain', '--untracked-files=all')
    if a.action == 'run' and (dirty or git('branch', '--show-current')):
        raise ValueError('Comparison requires clean detached execution snapshot')
    manifest_path = verified_file(c['manifest']); manifest = json.loads(manifest_path.read_text())
    for item in manifest['metadata']:
        verified_file(item)
    output.mkdir(parents=True, exist_ok=False); (output/'cache').mkdir()
    seed_all(c['seed']); torch.set_num_threads(4)
    module = GradientPrompting(build_model(c, a.device).sansa, c)
    model_hash = state_digest(module.sansa)
    provenance = dict(experiment='EXP-006',evidence='DEVELOPMENT' if a.action == 'run' else 'PILOT_ONLY', git_sha=git('rev-parse','HEAD'),
                      dirty=bool(dirty), config_sha256=sha256(config_path), manifest_sha256=sha256(manifest_path),
                      adapter=c['adapter'], sam2=c['sam2'], seed=c['seed'], torch=torch.__version__, cuda=torch.version.cuda,
                      cudnn=torch.backends.cudnn.version(), python=sys.version, platform=platform.platform(),
                      gpu=torch.cuda.get_device_name(a.device), device=a.device, command=sys.argv,
                      precision=c['precision'], start_unix=time.time(), pid=os.getpid())
    write(output/'provenance.json', provenance)
    entries = manifest['entries'][:a.limit] if a.limit else manifest['entries']
    checks = []
    for shots in c['shots']:
        with (output/f'paired_{shots}shot.jsonl').open('x', buffering=1) as log:
            for index, e in enumerate(entries):
                images, support, target = load_episode(e, shots, a.device)
                torch.cuda.reset_peak_memory_stats(a.device)
                cache, prep = timed(a.device, lambda: module.prepare(images, support))
                calls_before = module.calls
                outs, points, timing, stats, maps = generate(module, cache, c['seed']+index)
                if a.action == 'pilot':
                    from util.promptable_utils import build_prompt_dict
                    with torch.no_grad():
                        official = module.sansa(images, build_prompt_dict(support[None], 'mask', shots, False, torch.device(a.device)))['pred_masks'][-1]
                        equal = torch.equal(official, resized(outs['B0'], cache.size))
                        if not equal:
                            raise RuntimeError('Module-off official logits mismatch')
                        if points['S1'] is not None:
                            replay, timing['M1_nograd_audit'] = timed(a.device, lambda: module.decode(cache, points['S1']))
                            if not torch.equal(replay.low_res_masks, outs['S1'].low_res_masks):
                                raise RuntimeError('Graph-mode M1 differs from no-grad M1')
                        module.sansa.memory_bank = {}
                        # Re-running after the full refinement must use identical support state.
                        cache2 = module.prepare(images, support)
                        if not torch.equal(cache.feature, cache2.feature):
                            raise RuntimeError('Episode state leaked into repeated prepare')
                        checks.append(dict(shots=shots,episode_id=e['episode_id'],default_off_exact=True,replay_cache_exact=True,graph_nograd_m1_exact=True))
                        del cache2
                # All candidate generation above is complete before any query-label access below.
                rows = {}; lowres = dict(maps)
                previous = resized(outs['S1'], target.shape)
                for row, out in outs.items():
                    z = resized(out, target.shape)
                    rows[row] = dict(episode_id=e['episode_id'],query_id=e['query']['identity'],class_id=e['class_id'],
                                     counts=pixel_counts(z, target),boundary=boundary_counts(z, target),quality=module.quality(out),
                                     mask_unchanged_B0=bool(torch.equal(z>0, resized(outs['B0'],target.shape)>0)),
                                     mask_unchanged_S1=bool(torch.equal(z>0, previous>0)))
                    if row in points:
                        rows[row]['points'] = point_record(module, points[row], target, resized(outs['B0'],target.shape) if row=='S1' else previous)
                        rows[row]['valid_new_points'] = points[row] is not None
                        p0 = points.get('S1')
                        rows[row]['points_unchanged_S1'] = bool(points[row] is None and p0 is None) or bool(points[row] is not None and p0 is not None and torch.equal(points[row]['point_coords'], p0['point_coords']) and torch.equal(points[row]['point_labels'], p0['point_labels']))
                    lowres[row] = out.low_res_masks.detach().cpu().numpy()
                # Pure cache-based selection, with strict > retaining earliest ties as old MTP.
                for row in ['S1','S2','G2','G2shuffle','H2']:
                    names = ['B0','S1'] + ([] if row=='S1' else [row])
                    for suffix, score in [('quality','quality'), ('oracle',None)]:
                        key = max(names, key=lambda name: rows[name]['quality'] if score else episode_iou(rows[name]['counts']))
                        rows[row+'_'+suffix] = dict(rows[key],selected=key)
                diagnostics = {}
                if 'positive_domain' in maps:
                    gtlow = F.interpolate(target[None,None],outs['S1'].low_res_masks.shape[-2:],mode='nearest')[0,0].bool()
                    fn = (gtlow & (outs['S1'].low_res_masks[0,0] <= 0)).cpu().numpy()
                    stats['fn_pixels_lowres'] = int(fn.sum())
                    stats['fn_in_positive_domain'] = int((fn & maps['positive_domain']).sum())
                if e['gt_diagnostic']:
                    with torch.no_grad():
                        gtpoints, errors = gt_correction(module, outs['S1'], target, points['S1'])
                        gtout = outs['S1']
                        if points['S1'] is not None and gtpoints is not None:
                            gtout, timing['GT2'] = timed(a.device, lambda: module.decode(cache, gtpoints))
                        z = resized(gtout, target.shape)
                        diagnostics['GT2'] = dict(episode_id=e['episode_id'],query_id=e['query']['identity'],class_id=e['class_id'],
                            counts=pixel_counts(z,target),boundary=boundary_counts(z,target),
                            points=point_record(module,gtpoints,target,previous),error_domain_available=errors)
                        lowres['GT2'] = gtout.low_res_masks.detach().cpu().numpy()
                lowres['target'] = target.cpu().numpy().astype(np.uint8)
                np.savez_compressed(output/'cache'/f'{shots}shot_{index:03d}.npz', **lowres)
                timing['prepare'] = prep
                record = dict(index=index,episode_id=e['episode_id'],shots=shots,rows=rows,diagnostics=diagnostics,
                              stats=stats,timing=timing,head_calls=module.calls-calls_before,
                              peak_allocated_bytes=torch.cuda.max_memory_allocated(a.device))
                log.write(json.dumps(record,allow_nan=False)+'\n')
                del cache,outs,images,support,target
                if (index+1)%10 == 0 or index == 0:
                    print(json.dumps(dict(shots=shots,done=index+1,total=len(entries),stats=stats)),flush=True)
    if state_digest(module.sansa) != model_hash:
        raise RuntimeError('Frozen state changed')
    write(output/'checks.json',dict(frozen_model_unchanged=True,official_checks=checks))
    results = analyze(output,c['bootstrap_iterations'] if a.action=='run' else 100)
    write(output/'completed.json',dict(results_sha256=sha256(output/'results.json'), elapsed_seconds=time.time()-provenance['start_unix']))
    print(json.dumps({k:v['comparisons'] for k,v in results.items()}),flush=True)


if __name__ == '__main__':
    main()
