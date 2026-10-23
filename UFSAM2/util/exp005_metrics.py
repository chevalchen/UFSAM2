import math
import numpy as np
from scipy.ndimage import binary_erosion
from util.exp004_metrics import aggregate,bootstrap,episode_iou,pixel_counts,segmentation_loss


def boundary_counts(logits,target):
    p=(logits.sigmoid()>0.5).detach().cpu().numpy();y=(target==1).detach().cpu().numpy()
    radius=max(1,round(.02*math.hypot(*y.shape)))
    # Padding treats the image exterior as background, including objects touching its edge.
    def boundary(x):return x & ~binary_erosion(x,structure=np.ones((3,3)),iterations=radius,border_value=0)
    a,b=boundary(p),boundary(y)
    return [int((a&b).sum()),int((a|b).sum())]


def summarize(records):
    result=aggregate(records);classes={}
    for record in records:classes.setdefault(record['class_id'],[]).append(record)
    result['per_class']={str(k):aggregate(v) for k,v in classes.items()}
    if 'boundary' in records[0]:
        boundary={}
        for k,values in classes.items():
            total=np.sum([r['boundary'] for r in values],axis=0);boundary[str(k)]=100*float(total[0]/max(total[1],1))
        result['boundary_iou']=float(np.mean(list(boundary.values())))
        for k,value in boundary.items():result['per_class'][k]['boundary_iou']=value
    return result


def compare(rows):
    expected={'B0'}|{f'{r}/{s}' for r in ['G0','G1','G2'] for s in range(3)}
    if set(rows)!=expected:raise ValueError('Incomplete controls/seeds')
    identity=[(r['episode_id'],r['class_id']) for r in rows['B0']]
    for records in rows.values():
        if [(r['episode_id'],r['class_id']) for r in records]!=identity:raise ValueError('Unmatched episodes')
    metrics={key:summarize(value) for key,value in rows.items()};effects={}
    for control in ['B0','G0','G2']:
        gains=[];episode_effects=[]
        for seed in range(3):
            a=f'G1/{seed}';b=control if control=='B0' else f'{control}/{seed}'
            gains.append(metrics[a]['class_miou']-metrics[b]['class_miou'])
            episode_effects.append([episode_iou(x['counts'])-episode_iou(y['counts']) for x,y in zip(rows[a],rows[b])])
        ci=bootstrap(np.mean(episode_effects,axis=0))
        effects['G1-'+control]=dict(mean_class_miou_delta_pp=float(np.mean(gains)),seed_delta_pp=gains,seed_delta_sd=float(np.std(gains,ddof=1)),episode_fg_ci95_pp=ci,positive_ci=ci[0]>0)
    means={r:{k:dict(mean=float(np.mean([metrics[f'{r}/{s}'][k] for s in range(3)])),sd=float(np.std([metrics[f'{r}/{s}'][k] for s in range(3)],ddof=1))) for k in metrics['B0'] if k!='per_class'} for r in ['G0','G1','G2']}
    efficiency={}
    for key,records in rows.items():
        if 'decode_seconds' in records[0]:
            efficiency[key]=dict(mean_prepare_seconds=float(np.mean([r['prepare_seconds'] for r in records])),mean_decode_seconds=float(np.mean([r['decode_seconds'] for r in records])),peak_allocated_bytes=max(r['peak_allocated_bytes'] or 0 for r in records),diagnostics={k:float(np.mean([r['stats'][k] for r in records])) for k in records[0]['stats']})
    return dict(rows=metrics,seed_summary=means,comparisons=effects,efficiency=efficiency,episodes=len(identity),ci_scope='Paired episode FG effects averaged across seeds, not class-mIoU CI or seed uncertainty')


def decision(datasets):
    passing=[];safe=True
    for dataset,results in datasets.items():
        comparisons=[results['comparisons'][k] for k in ['G1-B0','G1-G0']]
        if all(c['mean_class_miou_delta_pp']>=.5 and c['positive_ci'] for c in comparisons):passing.append(dataset)
        safe &= all(c['mean_class_miou_delta_pp']>=-.2 for c in comparisons)
    return dict(continue_investigation=len(passing)>=2 and bool(safe),passing_datasets=passing,no_dataset_drop_over_point2=bool(safe),
                reliability_correspondence_supported={d:r['comparisons']['G1-G2']['mean_class_miou_delta_pp']>0 and r['comparisons']['G1-G2']['positive_ci'] for d,r in datasets.items()})
