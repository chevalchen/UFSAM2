"""Recompute candidate selections and compact diagnostics from EXP-006 caches."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'UFSAM2'))
from util.exp004_data import sha256
from util.exp004_metrics import episode_iou, pixel_counts
import torch
from torch.nn import functional as F


def main():
    p=argparse.ArgumentParser();p.add_argument('output');p.add_argument('--verify-cache',action='store_true');a=p.parse_args()
    out=Path(a.output);report={};compact=[];torch.set_num_threads(4)
    for shots in [1,5]:
        path=out/f'paired_{shots}shot.jsonl';records=[json.loads(s) for s in path.read_text().splitlines()]
        assert len(records)==120
        selections=0;cache_checks=0
        for r in records:
            rows=r['rows']
            for row in ['S1','S2','G2','G2shuffle','H2']:
                candidates=['B0','S1']+([] if row=='S1' else [row])
                for kind in ['quality','oracle']:
                    chosen=max(candidates,key=lambda key:rows[key]['quality'] if kind=='quality' else episode_iou(rows[key]['counts']))
                    assert rows[row+'_'+kind]['selected']==chosen
                    assert rows[row+'_'+kind]['counts']==rows[chosen]['counts']
                    selections+=1
            for row,v in rows.items():
                compact.append([shots,r['index'],r['episode_id'],v['query_id'],v['class_id'],row,*v['counts'],*v['boundary'],v['quality'],v.get('selected','')])
            if a.verify_cache:
                with np.load(out/'cache'/f"{shots}shot_{r['index']:03d}.npz") as cache:
                    target=torch.from_numpy(cache['target'])
                    for row in ['B0','S1','S2','G2','G2shuffle','H2']:
                        z=F.interpolate(torch.from_numpy(cache[row]),target.shape,mode='bilinear',align_corners=False)[0,0]
                        assert pixel_counts(z,target)==rows[row]['counts'],(shots,r['index'],row)
                        cache_checks+=1
        point_stats={}
        for row in ['S1','S2','G2','G2shuffle','H2']:
            values=[r['rows'][row] for r in records]
            by_label={}
            for lab in [1,0]:
                pairs=[(hit,error) for v in values for label,hit,error in zip(v['points']['labels'],v['points']['hits'],v['points']['error_hits']) if label==lab]
                by_label[str(lab)]=dict(points=len(pairs),hit_fraction=float(np.mean([p[0] for p in pairs])) if pairs else None,
                    error_hit_fraction=float(np.mean([p[1] for p in pairs])) if pairs else None)
            point_stats[row]=dict(valid_fraction=float(np.mean([v['valid_new_points'] for v in values])),
                point_count_histogram={str(n):sum(len(v['points']['labels'])==n for v in values) for n in [0,1,2]},
                points_unchanged_S1_fraction=float(np.mean([v['points_unchanged_S1'] for v in values])),
                mask_unchanged_B0_fraction=float(np.mean([v['mask_unchanged_B0'] for v in values])),
                mask_unchanged_S1_fraction=float(np.mean([v['mask_unchanged_S1'] for v in values])),by_label=by_label)
        effects={control:np.array([100*(episode_iou(r['rows']['G2']['counts'])-episode_iou(r['rows'][control]['counts'])) for r in records]) for control in ['B0','S2']}
        stats=[r['stats'] for r in records if r['stats']['gradient_valid']]
        fn=sum(s.get('fn_pixels_lowres',0) for s in stats);covered=sum(s.get('fn_in_positive_domain',0) for s in stats)
        cases={}
        for control in ['B0','S2']:
            for direction,idx in [('worst',int(effects[control].argmin())),('best',int(effects[control].argmax()))]:
                r=records[idx];cases[direction+'_G2-'+control]=dict(index=idx,episode_id=r['episode_id'],query_id=r['rows']['B0']['query_id'],
                    class_id=r['rows']['B0']['class_id'],delta_pp=float(effects[control][idx]),
                    episode_ious={key:100*episode_iou(r['rows'][key]['counts']) for key in ['B0','S1','S2','G2','G2shuffle','H2']})
        diagnostic=[r for r in records if 'GT2' in r['diagnostics']]
        gt_hits={str(lab):[(h,e) for r in diagnostic for l,h,e in zip(r['diagnostics']['GT2']['points']['labels'],r['diagnostics']['GT2']['points']['hits'],r['diagnostics']['GT2']['points']['error_hits']) if l==lab] for lab in [1,0]}
        report[str(shots)]=dict(point_stats=point_stats,selection_checks=selections,cache_count_checks=cache_checks,
            actual_query_head_calls=sum(r['head_calls'] for r in records),common_support_head_calls=shots*len(records),
            actual_backwards=len(stats),gradient_l2_range=[min(s['gradient_l2'] for s in stats),max(s['gradient_l2'] for s in stats)],
            mean_top_gradient_boundary_fraction=float(np.mean([s['gradient_top10_boundary_fraction'] for s in stats])),
            mean_boundary_image_fraction=float(np.mean([s['boundary_image_fraction'] for s in stats])),
            fn_pixels_lowres=fn,fn_in_positive_domain=covered,fn_coverage_fraction=covered/max(fn,1),
            identical_G2_S2_points_fraction=float(np.mean([r['rows']['G2']['points']['coords']==r['rows']['S2']['points']['coords'] for r in records])),
            tails={key:dict(drop_over_1pp=float((v < -1).mean()),drop_over_5pp=float((v < -5).mean())) for key,v in effects.items()},
            gt_points={k:dict(n=len(v),hit_fraction=float(np.mean([p[0] for p in v])),error_hit_fraction=float(np.mean([p[1] for p in v]))) for k,v in gt_hits.items()},
            cases=cases,paired_file_sha256=sha256(path))
    with (out/'analysis.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
    with (out/'paired_metrics.csv').open('x',newline='') as f:
        writer=csv.writer(f);writer.writerow(['shots','index','episode_id','query_id','class_id','row','bg_intersection','fg_intersection','bg_union','fg_union','boundary_intersection','boundary_union','quality','selected']);writer.writerows(compact)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
