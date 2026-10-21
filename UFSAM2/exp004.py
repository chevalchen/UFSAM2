"""EXP-004 pilot, equal-budget training, checkpoint locking and paired evaluation.

Run from repository root: python UFSAM2/exp004.py --help
"""
import argparse
import json
import time
from pathlib import Path

import torch

from models.sansa.local_prototype_residual import LocalPrototypeResidual
from util.exp004_data import ROOT, FixedEpisodes, load_contract, sha256
from util.exp004_metrics import aggregate, adjudicate, episode_iou, pixel_counts, segmentation_loss
from util.exp004_runtime import (assert_gradients, build_model, git, seed_all, start_run,
                                synchronize, write_json)


def load_episode(dataset, index, device):
    images, masks, target, entry = dataset[index]
    return images.to(device), masks.to(device), target.to(device), entry


def record(entry, logits, target):
    return dict(episode_id=entry['episode_id'], class_id=entry['class_id'], counts=pixel_counts(logits, target))


@torch.no_grad()
def evaluate(model, dataset, row, device):
    model.eval()
    records = []
    for index in range(len(dataset)):
        images, masks, target, entry = load_episode(dataset, index, device)
        cache = model.prepare(images, masks)
        records.append(record(entry, model.decode(cache, row), target))
    return records


def make_optimizer(model, contract):
    t = contract['training']
    return torch.optim.AdamW(model.residual.parameters(), lr=t['lr'], weight_decay=t['weight_decay'],
                             betas=tuple(t['betas']), eps=t['eps'])


def pilot(args, contract, manifest, output, provenance):
    seed_all(0)
    model = build_model(contract, args.device)
    dataset = FixedEpisodes(manifest, 'train', contract['data_root'])
    from util.promptable_utils import build_prompt_dict
    checks, timings = [], []
    if args.device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats(args.device)
    for index in range(contract['technical_pilot']['episodes']):
        images, masks, target, entry = load_episode(dataset, index, args.device)
        synchronize(args.device)
        start = time.perf_counter()
        cache = model.prepare(images, masks)
        synchronize(args.device)
        preparation = time.perf_counter() - start
        with torch.no_grad():
            baseline = model.sansa(images, build_prompt_dict(masks[None], 'mask', 5, False, torch.device(args.device)))['pred_masks'][-1]
            replay = model.decode(cache)
        binary_equal = torch.equal(baseline.sigmoid() > 0.5, replay.sigmoid() > 0.5)
        if not binary_equal:
            raise RuntimeError('Default-off differs from accepted SANSA')
        # Official forward saves memory_bank; release this diagnostic-only cache.
        model.sansa.memory_bank = {}
        row_checks = {}
        for row in ('C1', 'C2', 'C3'):
            seed_all(0)
            model.residual = LocalPrototypeResidual(cache.memory.shape[1]).to(args.device)
            optimizer = make_optimizer(model, contract)
            synchronize(args.device)
            context_start = time.perf_counter()
            context = model.context(cache, row)
            synchronize(args.device)
            context_seconds = time.perf_counter() - context_start
            with torch.no_grad():
                zero = model.decode(cache, row, context=context)
                if not torch.equal(replay, zero):
                    raise RuntimeError('Zero-init logits are not identical')
            initial = {k: v.clone() for k, v in model.residual.state_dict().items()}
            losses = []
            synchronize(args.device)
            start = time.perf_counter()
            model.train()
            for update in range(contract['technical_pilot']['steps_per_row']):
                optimizer.zero_grad(set_to_none=True)
                loss = segmentation_loss(model.decode(cache, row, context=context), target)
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite pilot loss')
                loss.backward()
                assert_gradients(model)
                if update > 0 and not any(torch.count_nonzero(p.grad) for p in model.residual.body.parameters()):
                    raise RuntimeError('Residual body has no nonzero gradient after output projection update')
                optimizer.step()
                losses.append(float(loss.detach()))
            synchronize(args.device)
            elapsed = time.perf_counter() - start
            changed = [k for k, v in model.residual.state_dict().items() if not torch.equal(initial[k], v)]
            if not changed or not any(k.startswith('body.') for k in changed):
                raise RuntimeError('Residual did not learn after two steps')
            row_checks[row] = dict(zero_init_exact=True, losses=losses, changed_tensors=len(changed),
                                   context_seconds=context_seconds,
                                   seconds_per_cached_training_step=elapsed / len(losses),
                                   estimated_online_step_seconds=preparation+context_seconds+elapsed/len(losses))
        timings.append(dict(episode_id=entry['episode_id'], prepare_seconds=preparation))
        checks.append(dict(episode_id=entry['episode_id'], default_off_binary_exact=binary_equal,
            baseline_logit_max_abs=float((baseline-replay).abs().max()), shape=list(cache.memory.shape), rows=row_checks))
    report = dict(evidence='DIAGNOSTIC_ONLY', checks=checks, timing=timings,
                  trainable_parameters=sum(p.numel() for p in model.residual.parameters()),
                  frozen_parameters=sum(p.numel() for p in model.sansa.parameters()),
                  peak_memory_allocated_bytes=torch.cuda.max_memory_allocated(args.device) if args.device.startswith('cuda') else None,
                  peak_memory_reserved_bytes=torch.cuda.max_memory_reserved(args.device) if args.device.startswith('cuda') else None)
    write_json(output / 'pilot.json', report)
    print(json.dumps(report, indent=2), flush=True)


def train(args, contract, manifest, output, provenance):
    if args.row not in contract['training']['rows'] or args.seed not in contract['training']['seeds']:
        raise ValueError('Unregistered row/seed')
    seed_all(args.seed)
    model = build_model(contract, args.device)
    # Reset after backbone construction to make all control initializations identical.
    seed_all(args.seed)
    model.residual = LocalPrototypeResidual(model.sansa.sam.hidden_dim).to(args.device)
    import hashlib
    initial_digest = hashlib.sha256()
    for name, value in model.residual.state_dict().items():
        initial_digest.update(name.encode())
        initial_digest.update(value.detach().cpu().numpy().tobytes())
    initial_sha256 = initial_digest.hexdigest()
    dataset = FixedEpisodes(manifest, 'train', contract['data_root'])
    calibration = FixedEpisodes(manifest, 'calibration', contract['data_root'])
    optimizer = make_optimizer(model, contract)
    best_score, best_step, checkpoints = -float('inf'), None, []
    start = time.perf_counter()
    if args.device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats(args.device)
    with (output / 'training.jsonl').open('x', buffering=1) as log:
        for step in range(1, contract['training']['steps'] + 1):
            model.train()
            images, masks, target, entry = load_episode(dataset, (step-1) % len(dataset), args.device)
            cache = model.prepare(images, masks)
            optimizer.zero_grad(set_to_none=True)
            logits = model.decode(cache, args.row)
            loss = segmentation_loss(logits, target)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training loss')
            loss.backward()
            assert_gradients(model)
            optimizer.step()
            log.write(json.dumps(dict(step=step, episode_id=entry['episode_id'], loss=float(loss.detach()))) + '\n')
            del cache, logits, loss
            if step % contract['training']['calibration_interval'] == 0:
                records = evaluate(model, calibration, args.row, args.device)
                metrics = aggregate(records)
                write_json(output / f'calibration_{step}.json', dict(records=records, metrics=metrics))
                checkpoint = output / f'residual_{step}.pt'
                torch.save(dict(state_dict=model.residual.state_dict(), row=args.row, seed=args.seed, step=step,
                                provenance=provenance), checkpoint)
                checkpoints.append(dict(step=step, path=str(checkpoint), sha256=sha256(checkpoint),
                                        calibration_class_miou=metrics['class_miou']))
                if metrics['class_miou'] > best_score:
                    best_score, best_step = metrics['class_miou'], step
                print(json.dumps(dict(row=args.row, seed=args.seed, step=step, calibration=metrics)), flush=True)
    chosen = next(c for c in checkpoints if c['step'] == best_step)
    write_json(output / 'completed.json', dict(row=args.row, seed=args.seed, steps=contract['training']['steps'],
        chosen=chosen, checkpoints=checkpoints, initial_sha256=initial_sha256,
        elapsed_seconds=time.perf_counter()-start, provenance=provenance,
        peak_memory_allocated_bytes=torch.cuda.max_memory_allocated(args.device) if args.device.startswith('cuda') else None))


def lock(args, contract, manifest):
    destination = Path(args.output).resolve()
    if destination == ROOT or ROOT in destination.parents:
        raise ValueError('Locked formal checkpoints must be external to repository')
    if len(args.runs) != 9:
        raise ValueError('Exactly nine completed runs required')
    chosen, seen, initial = [], set(), {}
    implementation = None
    for directory in args.runs:
        completion = Path(directory).resolve() / 'completed.json'
        done = json.loads(completion.read_text())
        key = (done['row'], done['seed'])
        if key in seen or key[0] not in ('C1', 'C2', 'C3') or key[1] not in (0, 1, 2):
            raise ValueError('Duplicate or wrong row/seed')
        seen.add(key)
        prov = done['provenance']
        if prov['mode'] != 'train' or prov['dirty'] or prov['contract_sha256'] != sha256(args.contract) or done['steps'] != 2400:
            raise ValueError('Invalid training provenance')
        if implementation is None:
            implementation = prov['implementation_sha']
        if implementation != prov['implementation_sha']:
            raise ValueError('Mixed implementation SHAs')
        if key[1] in initial and initial[key[1]] != done['initial_sha256']:
            raise ValueError('Controls did not share initialization')
        initial[key[1]] = done['initial_sha256']
        if [c['step'] for c in done['checkpoints']] != [600, 1200, 1800, 2400]:
            raise ValueError('Incomplete calibration schedule')
        for c in done['checkpoints']:
            if sha256(c['path']) != c['sha256']:
                raise ValueError('Checkpoint changed')
            cal = json.loads((completion.parent / f"calibration_{c['step']}.json").read_text())
            expected = manifest['partitions']['calibration']
            if [(e['episode_id'],e['class_id']) for e in cal['records']] != [(e['episode_id'],e['class_id']) for e in expected]:
                raise ValueError('Incomplete calibration episodes')
            if aggregate(cal['records'])['class_miou'] != c['calibration_class_miou']:
                raise ValueError('Calibration metric mismatch')
        best = max(done['checkpoints'], key=lambda c: (c['calibration_class_miou'], -c['step']))
        if done['chosen'] != best:
            raise ValueError('Wrong checkpoint selection')
        chosen.append(dict(row=key[0], seed=key[1], checkpoint=best,
                           completion_path=str(completion), completion_sha256=sha256(completion)))
    write_json(args.output, dict(experiment_id='EXP-004', contract_sha256=sha256(args.contract),
               implementation_sha=implementation, checkpoints=sorted(chosen,key=lambda x:(x['row'],x['seed']))))


@torch.no_grad()
def validate(args, contract, manifest, output, provenance):
    locked_path = Path(args.locked).resolve()
    locked = json.loads(locked_path.read_text())
    if locked['contract_sha256'] != provenance['contract_sha256'] or locked['implementation_sha'] != provenance['implementation_sha']:
        raise ValueError('Locked inputs differ from evaluation')
    expected = {(r, s) for r in ('C1', 'C2', 'C3') for s in range(3)}
    if len(locked['checkpoints']) != 9 or {(x['row'], x['seed']) for x in locked['checkpoints']} != expected:
        raise ValueError('Missing matched trained controls')
    modules = {}
    for item in locked['checkpoints']:
        if sha256(item['completion_path']) != item['completion_sha256'] or sha256(item['checkpoint']['path']) != item['checkpoint']['sha256']:
            raise ValueError('Locked artifact changed')
        saved = torch.load(item['checkpoint']['path'], map_location=args.device, weights_only=False)
        if (saved['row'], saved['seed'], saved['step']) != (item['row'], item['seed'], item['checkpoint']['step']):
            raise ValueError('Checkpoint identity mismatch')
        modules[f"{item['row']}/{item['seed']}"] = saved['state_dict']
    # Exclusive marker prevents an accidental second validation of this lock.
    write_json(str(locked_path) + '.validation_started.json', dict(output=str(output), lock_sha256=sha256(locked_path)))
    write_json(output / 'locked_checkpoints.json', locked)
    seed_all(0)
    model = build_model(contract, args.device).eval()
    dataset = FixedEpisodes(manifest, 'validation', contract['data_root'])
    rows = {'C0': []}
    rows.update({f'{r}/{s}': [] for r in ('C1','C2','C3','C4','C8') for s in range(3)})
    costs = []
    if args.device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats(args.device)
    with (output / 'paired.jsonl').open('x', buffering=1) as log:
        for index in range(len(dataset)):
            images, masks, target, entry = load_episode(dataset, index, args.device)
            synchronize(args.device)
            start = time.perf_counter()
            cache = model.prepare(images, masks)
            synchronize(args.device)
            prep = time.perf_counter()-start
            start = time.perf_counter()
            baseline = record(entry, model.decode(cache), target)
            synchronize(args.device)
            baseline_seconds = time.perf_counter()-start
            rows['C0'].append(baseline)
            episode = {'C0': baseline}
            latency = {}
            for seed in range(3):
                for row in ('C1','C2','C3','C4'):
                    model.residual.load_state_dict(modules[f"{'C3' if row == 'C4' else row}/{seed}"])
                    synchronize(args.device)
                    start = time.perf_counter()
                    context = model.context(cache, row)
                    synchronize(args.device)
                    context_time = time.perf_counter()-start
                    start = time.perf_counter()
                    result = record(entry, model.decode(cache, row, context=context), target)
                    synchronize(args.device)
                    decode_time = time.perf_counter()-start
                    key = f'{row}/{seed}'
                    rows[key].append(result)
                    episode[key] = result
                    latency[key] = dict(context_seconds=context_time, residual_decode_seconds=decode_time,
                        matched_path_seconds=prep+context_time+decode_time,
                        deploy_path_seconds=prep+decode_time+(0 if row=='C2' else context_time),
                        unused_context_seconds=context_time if row=='C2' else 0)
                treatment = episode[f'C3/{seed}']
                oracle = treatment if episode_iou(treatment['counts']) > episode_iou(baseline['counts']) else baseline
                rows[f'C8/{seed}'].append(oracle)
                episode[f'C8/{seed}'] = oracle
            costs.append(dict(episode_id=entry['episode_id'], preparation_seconds=prep,
                              baseline_decode_seconds=baseline_seconds, rows=latency))
            log.write(json.dumps(episode)+'\n')
            if (index+1) % 20 == 0:
                print(f'Validation {index+1}/{len(dataset)}', flush=True)
    result = adjudicate(rows)
    write_json(output / 'metrics.json', result)
    write_json(output / 'costs.json', dict(episodes=costs,
        note='Matched shared feature cache; C8 shares existing outputs, not an inference policy. Timing includes metric reduction; GPU synchronization at boundaries.',
        peak_memory_allocated_bytes=torch.cuda.max_memory_allocated(args.device) if args.device.startswith('cuda') else None))
    write_json(output / 'completed.json', dict(metrics_sha256=sha256(output/'metrics.json'),
               paired_sha256=sha256(output/'paired.jsonl'), costs_sha256=sha256(output/'costs.json'), provenance=provenance))
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['pilot', 'train', 'lock', 'validate'])
    parser.add_argument('--contract', default=str(ROOT / 'experiments/EXP-004/configs/contract_v1.json'))
    parser.add_argument('--execution', help='Tracked signed authorization JSON, required for formal runs')
    parser.add_argument('--output', required=True, help='New immutable directory; lock mode: new JSON file')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--row', choices=['C1','C2','C3'])
    parser.add_argument('--seed', type=int, choices=[0,1,2], default=0)
    parser.add_argument('--runs', nargs='*', default=[])
    parser.add_argument('--locked')
    args = parser.parse_args()
    torch.set_num_threads(4)
    contract, manifest = load_contract(args.contract)
    if args.mode == 'lock':
        lock(args, contract, manifest)
        return
    output, provenance = start_run(args, contract, args.mode)
    try:
        globals()[args.mode](args, contract, manifest, output, provenance)
    except BaseException as error:
        write_json(output / 'failed.json', dict(status='INCONCLUSIVE', error=repr(error), provenance=provenance))
        raise

if __name__ == '__main__':
    main()
