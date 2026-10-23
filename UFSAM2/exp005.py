"""EXP-005: pilot, matched training, checkpoint lock and three-dataset evaluation."""
import argparse,hashlib,json,subprocess,time,sys
from pathlib import Path
import torch
from models.sansa.support_graph import GraphSANSA,SupportGraph
from util.exp004_runtime import seed_all,build_model,synchronize,write_json
from util.exp004_data import sha256
from util.exp005_data import ROOT,Episodes,load_inputs
from util.exp005_metrics import pixel_counts,segmentation_loss,boundary_counts,summarize,compare,decision


def build(c,device,seed):
    seed_all(seed);sansa=build_model(c,device).sansa
    seed_all(seed);return GraphSANSA(sansa)


def digest(module):
    h=hashlib.sha256()
    for name,value in module.state_dict().items():h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def gradients(model,strict=False):
    if any(p.requires_grad or p.grad is not None for p in model.sansa.parameters()):raise RuntimeError('Frozen SANSA gradient/parameter violation')
    for name,p in model.graph.named_parameters():
        if p.grad is None or not torch.isfinite(p.grad).all():raise RuntimeError('Invalid gradient '+name)
    if strict:
        for module in [model.graph.input,*model.graph.layers,model.graph.mu,model.graph.var,model.graph.output]:
            if not any(torch.count_nonzero(p.grad) for p in module.parameters()):raise RuntimeError('Disconnected graph branch')


def episode(ds,index,device):
    images,masks,target,e=ds[index];return images.to(device),masks.to(device),target.to(device),e


def rec(e,logits,target,parts=False):
    r=dict(episode_id=e['episode_id'],class_id=e['class_id'],counts=pixel_counts(logits,target))
    if parts:r['boundary']=boundary_counts(logits,target)
    return r


def optimizer(model,c):return torch.optim.AdamW(model.graph.parameters(),lr=c['lr'],weight_decay=c['weight_decay'])


@torch.no_grad()
def evaluate(model,ds,mode,device):
    model.eval();records=[]
    for i in range(len(ds)):
        images,masks,target,e=episode(ds,i,device);cache=model.prepare(images,masks)
        logits,_,_,_=model.decode(cache,mode,e['episode_id'],auxiliary=False)
        records.append(rec(e,logits,target))
    return records


def pilot(a,c,m,out,prov):
    from util.promptable_utils import build_prompt_dict
    model=build(c,a.device,0);base_hash=digest(model.sansa);reports=[]
    if a.device.startswith('cuda'):torch.cuda.reset_peak_memory_stats(a.device)
    for dataset in ['train','pascal_part','paco_part']:
        ds=Episodes(m,dataset);images,masks,target,e=episode(ds,0,a.device)
        synchronize(a.device);start=time.perf_counter();cache=model.prepare(images,masks);synchronize(a.device);prep=time.perf_counter()-start
        with torch.no_grad():
            accepted=model.sansa(images,build_prompt_dict(masks[None],'mask',5,False,torch.device(a.device)))['pred_masks'][-1]
            baseline=model.decode(cache,'baseline',e['episode_id'])[0]
            if not torch.equal(accepted,baseline):raise RuntimeError('Default-off logit mismatch')
            model.sansa.memory_bank={}
        for mode in ['uniform','uncertainty']:
            seed_all(0);model.graph=SupportGraph().to(a.device);model.eval()
            with torch.no_grad():
                zero=model.decode(cache,mode,e['episode_id'])[0]
                if not torch.equal(zero,baseline):raise RuntimeError('Zero-init mismatch')
            losses=[];opt=optimizer(model,c);model.train();start=time.perf_counter()
            for step in range(2):
                opt.zero_grad(set_to_none=True)
                z,aux,stats,bank=model.decode(cache,mode,e['episode_id'],step,retain_memory=True)
                seg=segmentation_loss(z,target)
                # Verify the segmentation loss alone reaches graph heads on the second step.
                if step==1:
                    seg.backward(retain_graph=True);gradients(model,strict=True);opt.zero_grad(set_to_none=True)
                loss=seg+aux
                loss.backward();gradients(model,strict=step==1)
                if any(entry['maskmem_features'].grad is None or not torch.isfinite(entry['maskmem_features'].grad).all() or not torch.count_nonzero(entry['maskmem_features'].grad) for entry in bank.values()):raise RuntimeError('A support memory has no segmentation gradient')
                opt.step();losses.append(float(loss.detach()))
            synchronize(a.device)
            reports.append(dict(dataset=dataset,mode=mode,default_off_exact=True,zero_init_exact=True,all_five_memory_gradients=True,segmentation_only_graph_gradients=True,losses=losses,prepare_seconds=prep,two_steps_seconds=time.perf_counter()-start,stats=stats))
    if digest(model.sansa)!=base_hash:raise RuntimeError('Frozen SANSA state changed')
    write_json(out/'pilot.json',dict(evidence='DIAGNOSTIC_ONLY',checks=reports,frozen_state_unchanged=True,trainable_parameters=sum(p.numel() for p in model.graph.parameters()),peak_allocated_bytes=torch.cuda.max_memory_allocated(a.device) if a.device.startswith('cuda') else None))
    print('Pilot PASS',flush=True)


def train(a,c,m,out,prov):
    model=build(c,a.device,a.seed);initial=digest(model.graph);opt=optimizer(model,c)
    ds=Episodes(m,'train');cal=Episodes(m,'calibration');saved=[]
    if a.device.startswith('cuda'):torch.cuda.reset_peak_memory_stats(a.device)
    start=time.perf_counter()
    with (out/'training.jsonl').open('x',buffering=1) as log:
        for step in range(1,c['steps']+1):
            model.train();images,masks,target,e=episode(ds,(step-1)%len(ds),a.device)
            cache=model.prepare(images,masks);opt.zero_grad(set_to_none=True)
            z,aux,stats,_=model.decode(cache,a.mode,e['episode_id'],a.seed*100000+step)
            seg=segmentation_loss(z,target);loss=seg+aux
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            loss.backward();gradients(model,strict=step==2);opt.step()
            if any(not torch.isfinite(p).all() for p in model.graph.parameters()):raise RuntimeError('Nonfinite parameters')
            log.write(json.dumps(dict(step=step,episode_id=e['episode_id'],loss=float(loss.detach()),segmentation_loss=float(seg.detach()),auxiliary_loss=float(aux.detach()),**stats))+'\n')
            del cache,z,loss,seg,aux
            if step%c['calibration_interval']==0:
                records=evaluate(model,cal,a.mode,a.device);metrics=summarize(records)
                write_json(out/f'calibration_{step}.json',dict(records=records,metrics=metrics))
                path=out/f'graph_{step}.pt';torch.save(dict(state_dict=model.graph.state_dict(),mode=a.mode,seed=a.seed,step=step,provenance=prov),path)
                saved.append(dict(step=step,path=str(path),sha256=sha256(path),score=metrics['class_miou']))
                print(json.dumps(dict(step=step,mode=a.mode,seed=a.seed,calibration_miou=metrics['class_miou'])),flush=True)
    chosen=max(saved,key=lambda s:(s['score'],-s['step']))
    write_json(out/'completed.json',dict(mode=a.mode,seed=a.seed,initial_sha256=initial,checkpoints=saved,chosen=chosen,steps=c['steps'],elapsed_seconds=time.perf_counter()-start,provenance=prov,peak_allocated_bytes=torch.cuda.max_memory_allocated(a.device) if a.device.startswith('cuda') else None))


def lock_runs(paths,c,m,config_hash,code_sha):
    seen=set();initial={};locked=[]
    for directory in paths:
        p=Path(directory)/'completed.json';d=json.loads(p.read_text());key=(d['mode'],d['seed'])
        if key in seen or key[0] not in ['uniform','uncertainty'] or key[1] not in [0,1,2]:raise ValueError('Invalid/duplicate run')
        seen.add(key)
        if d['steps']!=2400 or d['provenance']['config_sha256']!=config_hash or d['provenance']['git_sha']!=code_sha or d['provenance']['dirty']:raise ValueError('Training provenance mismatch')
        initial.setdefault(key[1],d['initial_sha256'])
        if initial[key[1]]!=d['initial_sha256']:raise ValueError('Unmatched initialization')
        records=[json.loads(line) for line in (p.parent/'training.jsonl').read_text().splitlines()]
        if len(records)!=2400 or any(r['step']!=i+1 or r['episode_id']!=m['partitions']['train'][i%1200]['episode_id'] for i,r in enumerate(records)):raise ValueError('Training exposure mismatch')
        if [v['step'] for v in d['checkpoints']]!=[600,1200,1800,2400]:raise ValueError('Missing calibration')
        for ck in d['checkpoints']:
            if sha256(ck['path'])!=ck['sha256']:raise ValueError('Checkpoint changed')
            cal=json.loads((p.parent/f"calibration_{ck['step']}.json").read_text())
            if [(x['episode_id'],x['class_id']) for x in cal['records']]!=[(x['episode_id'],x['class_id']) for x in m['partitions']['calibration']]:raise ValueError('Calibration coverage mismatch')
            if summarize(cal['records'])['class_miou']!=ck['score']:raise ValueError('Calibration score mismatch')
        if d['chosen']!=max(d['checkpoints'],key=lambda s:(s['score'],-s['step'])):raise ValueError('Invalid checkpoint selection')
        locked.append(dict(mode=key[0],seed=key[1],checkpoint=d['chosen'],completion_sha256=sha256(p)))
    if len(seen)!=6:raise ValueError('Six matched runs required')
    return locked


@torch.no_grad()
def validate(a,c,m,out,prov):
    locked=lock_runs(a.runs,c,m,prov['config_sha256'],prov['git_sha']);write_json(out/'locked.json',locked)
    model=build(c,a.device,0).eval();states={}
    for item in locked:
        saved=torch.load(item['checkpoint']['path'],map_location=a.device,weights_only=False)
        if (saved['mode'],saved['seed'],saved['step'])!=(item['mode'],item['seed'],item['checkpoint']['step']):raise ValueError('Checkpoint identity mismatch')
        states[(item['mode'],item['seed'])]=saved['state_dict']
    results={}
    for name in ['coco','pascal_part','paco_part']:
        ds=Episodes(m,name);rows={'B0':[]};rows.update({f'{r}/{s}':[] for r in ['G0','G1','G2'] for s in range(3)})
        with (out/f'{name}_paired.jsonl').open('x',buffering=1) as log:
            for i in range(len(ds)):
                images,masks,target,e=episode(ds,i,a.device)
                if a.device.startswith('cuda'):torch.cuda.reset_peak_memory_stats(a.device)
                synchronize(a.device);start=time.perf_counter();cache=model.prepare(images,masks);synchronize(a.device);prep=time.perf_counter()-start
                episode_results={}
                for key in rows:
                    if key=='B0':mode='baseline'
                    else:
                        row,seed=key.split('/');mode={'G0':'uniform','G1':'uncertainty','G2':'shuffled'}[row]
                        model.graph.load_state_dict(states[('uniform' if row=='G0' else 'uncertainty',int(seed))])
                    if a.device.startswith('cuda'):torch.cuda.reset_peak_memory_stats(a.device)
                    synchronize(a.device);start=time.perf_counter();z,_,stats,_=model.decode(cache,mode,e['episode_id'],auxiliary=False);synchronize(a.device);seconds=time.perf_counter()-start
                    record=rec(e,z,target,name!='coco');record.update(stats=stats,decode_seconds=seconds,prepare_seconds=prep,peak_allocated_bytes=torch.cuda.max_memory_allocated(a.device) if a.device.startswith('cuda') else None)
                    rows[key].append(record);episode_results[key]=record
                log.write(json.dumps(episode_results)+'\n')
                if (i+1)%50==0:print(name,i+1,len(ds),flush=True)
        results[name]=compare(rows);write_json(out/f'{name}_metrics.json',results[name])
    report=dict(datasets=results,decision=decision(results),provenance=prov)
    write_json(out/'results.json',report)
    lines=['# EXP-005 三数据集结果', '', '| 数据集 | B0 class-mIoU | G0 mean ± SD | G1 mean ± SD | G2 mean ± SD |', '|---|---:|---:|---:|---:|']
    for dataset,result in results.items():
        cells=[dataset, f"{result['rows']['B0']['class_miou']:.4f}"]
        for row in ['G0','G1','G2']:
            value=result['seed_summary'][row]['class_miou'];cells.append(f"{value['mean']:.4f} ± {value['sd']:.4f}")
        lines.append('| '+' | '.join(cells)+' |')
    lines.extend(['', '规则计算结果：'+json.dumps(report['decision'],ensure_ascii=False), '', 'CI 及各类结果见 results.json；本表不是 novel-class benchmark 结论。'])
    with (out/'results.md').open('x') as handle:handle.write('\n'.join(lines)+'\n')
    write_json(out/'completed.json',dict(results_sha256=sha256(out/'results.json'),decision=report['decision']))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['pilot','train','validate'])
    p.add_argument('--config',default=str(ROOT/'experiments/EXP-005/config.json'));p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--mode',choices=['uniform','uncertainty'],default='uncertainty');p.add_argument('--seed',type=int,choices=[0,1,2],default=0);p.add_argument('--runs',nargs='*',default=[])
    a=p.parse_args();torch.set_num_threads(4);c,m=load_inputs(a.config)
    dirty=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    if a.action!='pilot' and dirty:raise ValueError('Commit implementation before training; no remote handoff needed')
    out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
    prov=dict(experiment='EXP-005',action=a.action,git_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),dirty=bool(dirty),config_sha256=sha256(a.config),manifest_sha256=c['manifest']['sha256'],adapter_sha256=c['adapter']['sha256'],sam2_sha256=c['sam2']['sha256'],torch=torch.__version__,cuda=torch.version.cuda,command=sys.argv,start_unix=time.time())
    write_json(out/'provenance.json',prov)
    try:globals()[a.action](a,c,m,out,prov)
    except BaseException as error:
        write_json(out/'failed.json',dict(error=repr(error),provenance=prov));raise

if __name__=='__main__':main()
