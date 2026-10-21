"""Execution-only EXP-003 queue: wait for idle CUDA, validate smoke, run Stage A.

Writes external execution facts only. Never changes research decisions or Git.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args):
    return subprocess.check_output(['git', *args], text=True).strip()


def available_gpu():
    rows = subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,uuid,memory.free,utilization.gpu',
        '--format=csv,noheader,nounits',
    ], text=True).splitlines()
    busy = subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader',
    ], text=True).splitlines()
    for row in rows:
        index, uuid, free, usage = [v.strip() for v in row.split(',')]
        if uuid not in busy and int(free) >= 22000 and int(usage) <= 5:
            return index
    return None


def validate_result(directory, config, expected_sha, diagnostic):
    metrics = json.loads((directory / 'stage_a_leave_one_out_metrics.json').read_text())
    manifest = json.loads((directory / 'run_manifest.json').read_text())
    records = json.loads((directory / 'stage_a_leave_one_out_per_episode.json').read_text())['records']
    count = 1 if diagnostic else 600
    if metrics['git_sha'] != expected_sha or manifest['git']['commit'] != expected_sha:
        raise RuntimeError('Execution SHA mismatch')
    if not manifest['git']['clean'] or manifest['git']['branch'] is not None:
        raise RuntimeError('Execution worktree was not clean and detached')
    if metrics['evidence_label'] != ('DIAGNOSTIC_ONLY' if diagnostic else 'FORMAL'):
        raise RuntimeError('Evidence label mismatch')
    if metrics['episode_count'] != count or metrics['unique_episode_ids'] != count or len(records) != count:
        raise RuntimeError('Episode count mismatch')
    if metrics['loaded_adapter_tensors'] != 240:
        raise RuntimeError('Adapter load mismatch')
    for field, expected in {
        'execution_config_sha256': sha256(config),
        'episode_manifest_sha256': manifest['inputs']['episodes']['sha256'],
        'adapter_sha256': manifest['inputs']['checkpoint']['sha256'],
        'sam2_checkpoint_sha256': manifest['inputs']['sam2_base']['sha256'],
        'per_episode_sha256': sha256(directory / 'stage_a_leave_one_out_per_episode.json'),
    }.items():
        if metrics[field] != expected:
            raise RuntimeError(f'Provenance mismatch: {field}')
    if diagnostic:
        checks = metrics['cached_reference_checks']
        expected_rows = {'B0', 'D0', 'D1', 'D2', 'D3', 'D4'}
        if len(checks) != 1 or not checks[0]['passed'] or set(checks[0]['rows']) != expected_rows:
            raise RuntimeError('Missing six-way reference equivalence')
        if not all(row['binary_mask_exact'] for row in checks[0]['rows'].values()):
            raise RuntimeError('Cached/reference prediction mismatch')
    return metrics, records


def main(args):
    control = Path(args.control_dir)
    control.mkdir(parents=True, exist_ok=True)
    lock = (control / 'queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (control / 'status.json').exists():
        raise FileExistsError('Queue already has a status; use a new control directory')
    config_path = Path(args.execution_config).resolve()
    config = json.loads(config_path.read_text())
    expected_sha = git('rev-parse', 'HEAD')

    def status(state, **extra):
        value = {'state': state, 'updated_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                 'git_sha': expected_sha, 'pid': os.getpid(), **extra}
        temporary = control / 'status.tmp'
        temporary.write_text(json.dumps(value, indent=2) + '\n')
        temporary.replace(control / 'status.json')
        print(json.dumps(value), flush=True)

    def clean():
        if git('status', '--porcelain') or git('branch', '--show-current'):
            raise RuntimeError('Queue requires a clean detached worktree')
        if git('rev-parse', 'HEAD') != expected_sha:
            raise RuntimeError('Worktree SHA changed')

    def execute(argv, logname, env):
        with (control / logname).open('x') as log:
            subprocess.run(argv, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)

    try:
        clean()
        if config.get('formal_execution_authorized') is not True:
            raise RuntimeError('Formal execution has not been authorized')
        status('WAITING_FOR_IDLE_GPU', minimum_free_mib=22000)
        while True:
            if (control / 'STOP').exists():
                status('CANCELLED_BEFORE_EXECUTION')
                return
            selected = available_gpu()
            time.sleep(30)
            if selected is not None and available_gpu() == selected:
                break
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=selected, OMP_NUM_THREADS='4',
                   MKL_NUM_THREADS='4', PYTHONPATH=str(Path('UFSAM2').resolve()),
                   MPLCONFIGDIR=str(control / 'matplotlib'))
        clean()
        status('GPU_TESTS_RUNNING', gpu=selected)
        execute([sys.executable, '-m', 'unittest', 'discover', '-s', 'UFSAM2/tests',
                 '-t', 'UFSAM2', '-v'], 'tests.log', env)
        smoke_records = None
        for diagnostic in (True, False):
            clean()
            section = config['execution']
            packet = section['diagnostic_smoke'] if diagnostic else section
            argv = packet['argv' if diagnostic else 'formal_evaluation_argv']
            argv = [sys.executable] + argv[argv.index('python') + 1:]
            run_dir = Path(argv[argv.index('--output-dir') + 1]) / argv[argv.index('--name-exp') + 1]
            capture = packet['run_manifest_argv' if diagnostic else 'formal_run_manifest_argv'].copy()
            capture[0] = sys.executable
            capture[capture.index('--command') + 1] = shlex.join(argv)
            label = 'smoke' if diagnostic else 'formal'
            status('GPU_SMOKE_RUNNING' if diagnostic else 'FORMAL_RUNNING', gpu=selected, run_dir=str(run_dir))
            execute(capture, f'{label}_manifest.log', env)
            execute(argv, f'{label}_console.log', env)
            metrics, records = validate_result(run_dir, config_path, expected_sha, diagnostic)
            if diagnostic:
                smoke_records = records
            elif records[0]['primitive_fingerprints'] != smoke_records[0]['primitive_fingerprints']:
                raise RuntimeError('Smoke/formal first-episode primitive fingerprints differ')
        report = {'evidence_label': 'UNADJUDICATED_EXECUTION_REPORT', 'git_sha': expected_sha,
                  'metrics_path': str(run_dir / 'stage_a_leave_one_out_metrics.json'),
                  'metrics_sha256': sha256(run_dir / 'stage_a_leave_one_out_metrics.json'),
                  'rows': metrics['rows'], 'effects': metrics['effects'],
                  'guardrails': metrics['guardrails'], 'supporting': metrics['supporting'],
                  'automatic_stage_suggestion': metrics['automatic_stage_suggestion']}
        (control / 'execution_report.json').write_text(json.dumps(report, indent=2) + '\n')
        status('COMPLETED_UNADJUDICATED', gpu=selected, report=str(control / 'execution_report.json'))
    except Exception as exc:
        status('FAILED', error=f'{type(exc).__name__}: {exc}')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--execution-config', required=True)
    parser.add_argument('--control-dir', required=True)
    main(parser.parse_args())
