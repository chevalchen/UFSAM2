"""Independent warm cached-head cost measurements, excluding diagnostics and metrics."""
import argparse
import json
from pathlib import Path
import time
import torch
from exp006 import (ROOT, git, load_episode, timed, write, state_digest)
from util.exp004_data import verified_file, sha256
from util.exp004_runtime import build_model, seed_all, synchronize
from models.sansa.gradient_prompting import GradientPrompting


def pipeline(module,cache,row):
    backward=0
    with torch.no_grad():
        out=module.decode(cache)
        if row=='B0':return backward
        p=module.points(out)
        if p is None:return backward
    if row=='G2':
        feature=cache.feature.clone().requires_grad_(True)
        out=module.decode(cache,p,feature)
        response,stats=module.gradient(feature,out)
        backward=int(stats['gradient_valid'])
        out=module.detach(out)
    else:
        with torch.no_grad():out=module.decode(cache,p)
    if row=='S1':return backward
    with torch.no_grad():
        p=module.points(out,p,response if row=='G2' else None)
        if p is not None:module.decode(cache,p)
    return backward


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:1');a=p.parse_args()
    if git('status','--porcelain') or git('branch','--show-current'):raise RuntimeError('Clean detached snapshot required')
    c=json.loads((ROOT/'experiments/EXP-006/config_v1.json').read_text());manifest=json.loads(verified_file(c['manifest']).read_text())
    output=Path(a.output);output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);seed_all(c['seed'])
    module=GradientPrompting(build_model(c,a.device).sansa,c);before=state_digest(module.sansa);records=[]
    # Eight predetermined queries spanning classes; no result-driven selection.
    for shots in [1,5]:
        for i,index in enumerate(range(0,120,15)):
            e=manifest['entries'][index];images,support,_=load_episode(e,shots,a.device)
            torch.cuda.reset_peak_memory_stats(a.device)
            cache,prep=timed(a.device,lambda:module.prepare(images,support));prep_peak=torch.cuda.max_memory_allocated(a.device)
            # Warm each path once, then cycle row order to reduce systematic shared-GPU drift.
            for row in ['B0','S1','S2','G2']:pipeline(module,cache,row)
            for repeat in range(3):
                order=['B0','S1','S2','G2'];shift=(i+repeat)%4;order=order[shift:]+order[:shift]
                for row in order:
                    synchronize(a.device);torch.cuda.reset_peak_memory_stats(a.device);calls=module.calls
                    backwards,seconds=timed(a.device,lambda:pipeline(module,cache,row))
                    records.append(dict(shots=shots,index=index,row=row,repeat=repeat,head_calls=module.calls-calls,
                        backwards=backwards,cached_query_seconds=seconds,prepare_seconds=prep,
                        head_peak_allocated_bytes=torch.cuda.max_memory_allocated(a.device),prepare_peak_allocated_bytes=prep_peak))
            del cache,images,support
    assert state_digest(module.sansa)==before
    import numpy as np
    summary={str(shots):{row:dict(mean_query_ms=1000*float(np.mean([r['cached_query_seconds'] for r in records if r['shots']==shots and r['row']==row])),
        median_query_ms=1000*float(np.median([r['cached_query_seconds'] for r in records if r['shots']==shots and r['row']==row])),
        peak_head_allocated_gib=max(r['head_peak_allocated_bytes'] for r in records if r['shots']==shots and r['row']==row)/2**30,
        peak_prepare_allocated_gib=max(r['prepare_peak_allocated_bytes'] for r in records if r['shots']==shots and r['row']==row)/2**30,
        mean_prepare_ms=1000*float(np.mean([r['prepare_seconds'] for r in records if r['shots']==shots and r['row']==row]))) for row in ['B0','S1','S2','G2']} for shots in [1,5]}
    write(output/'profile.json',dict(git_sha=git('rev-parse','HEAD'),seed=c['seed'],config_sha256=sha256(ROOT/'experiments/EXP-006/config_v1.json'),
        device=a.device,gpu=torch.cuda.get_device_name(a.device),torch=torch.__version__,command=__import__('sys').argv,
        scope='8 predetermined queries per shot, 3 rotated-order warm repeats; cached query includes point sampling; shared GPU',
        records=records,summary=summary,frozen_model_unchanged=True))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
