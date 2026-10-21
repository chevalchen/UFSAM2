"""Strict EXP-004 input binding, frozen SANSA loading and run provenance."""
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from util.exp004_data import ROOT, sha256, verified_file


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(contract, device):
    from models.sansa.sansa import build_sansa
    from models.sansa.local_prototype_residual import FrozenSANSARepair
    from util.checkpoint_validation import validate_sansa_base_state
    adapter, sam2 = verified_file(contract['adapter']), verified_file(contract['sam2'])
    model = build_sansa(sam2_version='large', adaptformer_stages=[2, 3], channel_factor=0.3,
                        device=device, sam2_checkpoint=str(sam2))
    saved = torch.load(adapter, map_location='cpu', weights_only=False)
    state, keys = validate_sansa_base_state(model.state_dict(), saved.get('model', saved))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if set(keys) & set(missing) or unexpected:
        raise ValueError('Adapter loading failed')
    model.to(device).requires_grad_(False).eval()
    return FrozenSANSARepair(model)


def write_json(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def start_run(args, contract, mode):
    """Formal runs require a separately signed, tracked execution authorization."""
    config = Path(args.contract).resolve()
    commit = git('rev-parse', 'HEAD')
    dirty = git('status', '--porcelain', '--untracked-files=all')
    if mode != 'pilot':
        if not args.execution:
            raise ValueError('Formal execution needs a signed --execution JSON; implementation is not authorization')
        execution_path = Path(args.execution).resolve()
        execution = json.loads(execution_path.read_text())
        if dirty or git('branch', '--show-current'):
            raise ValueError('Formal run requires a clean detached exact-SHA worktree')
        for p in (config, execution_path, verified_file(contract['manifest']), verified_file(contract['exclusions'])):
            git('ls-files', '--error-unmatch', str(p.relative_to(ROOT)))
        # execution SHA is the implementation ancestor; authorization commit may follow it.
        if execution.get('experiment_id') != 'EXP-004' or execution.get('authorized') is not True:
            raise ValueError('Formal execution is not authorized')
        if execution.get('contract_sha256') != sha256(config):
            raise ValueError('Execution contract hash mismatch')
        signed = execution['implementation_sha']
        git('merge-base', '--is-ancestor', signed, commit)
        if git('diff', signed, commit, '--', 'UFSAM2', 'scripts', 'experiments/EXP-004/configs/contract_v1.json', 'experiments/EXP-004/episodes'):
            raise ValueError('Code or scientific inputs differ from signed implementation')
    output = Path(args.output).resolve()
    if mode != 'pilot' and (output == ROOT or ROOT in output.parents):
        raise ValueError('Formal artifacts must be external to repository')
    output.mkdir(parents=True, exist_ok=False)
    provenance = dict(experiment_id='EXP-004', mode=mode,
        evidence='DIAGNOSTIC_ONLY' if mode == 'pilot' else 'UNADJUDICATED_EXECUTION',
        git_sha=commit, dirty=bool(dirty), contract_sha256=sha256(config),
        manifest_sha256=contract['manifest']['sha256'], adapter_sha256=contract['adapter']['sha256'],
        sam2_sha256=contract['sam2']['sha256'], command=sys.argv,
        utc=datetime.now(timezone.utc).isoformat(), python=sys.version, torch=torch.__version__,
        cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(), platform=platform.platform(),
        device=args.device, cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        precision=contract['precision'])
    if mode != 'pilot':
        provenance['execution_sha256'] = sha256(execution_path)
        provenance['implementation_sha'] = execution['implementation_sha']
        # Required neutral capture script independently enforces clean/tracked inputs.
        subprocess.run([sys.executable, str(ROOT / '.agents/skills/manage-ufs-research/scripts/capture_run_manifest.py'),
            '--experiment', 'EXP-004', '--config', str(config),
            '--episodes', str(verified_file(contract['manifest'])), '--checkpoint', contract['adapter']['path'],
            '--seed', str(args.seed if hasattr(args, 'seed') else 0), '--fold', '0',
            '--command', repr(sys.argv), '--output', str(output / 'manifest.json')], cwd=ROOT, check=True)
    write_json(output / 'provenance.json', provenance)
    return output, provenance


def synchronize(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize(device)


def assert_gradients(model):
    if any(p.requires_grad or p.grad is not None for p in model.sansa.parameters()):
        raise RuntimeError('Frozen SANSA unexpectedly trainable or has gradients')
    grads = [p.grad for p in model.residual.parameters()]
    if any(g is None or not torch.isfinite(g).all() for g in grads) or not any(torch.count_nonzero(g) for g in grads):
        raise RuntimeError('Missing, non-finite or all-zero residual gradients')
