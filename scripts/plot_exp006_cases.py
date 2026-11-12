"""Render measured best/worst and GT-repair cases from fixed candidate caches."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'UFSAM2'))
from util.exp004_metrics import episode_iou


def main():
    p=argparse.ArgumentParser();p.add_argument('output');a=p.parse_args();out=Path(a.output)
    entries=json.loads((ROOT/'experiments/EXP-006/episodes_v1.json').read_text())['entries']
    figure_dir=out/'figures';figure_dir.mkdir(exist_ok=False);selected=[]
    torch.set_num_threads(4)
    for shots in [1,5]:
        records=[json.loads(line) for line in (out/f'paired_{shots}shot.jsonl').read_text().splitlines()]
        delta=np.array([episode_iou(r['rows']['G2']['counts'])-episode_iou(r['rows']['S2']['counts']) for r in records])
        diagnostic=[r for r in records if 'GT2' in r['diagnostics']]
        gtbest=max(diagnostic,key=lambda r:episode_iou(r['diagnostics']['GT2']['counts'])-episode_iou(r['rows']['S1']['counts']))
        cases=[('worst_G2_S2',int(delta.argmin())),('best_G2_S2',int(delta.argmax())),('largest_GT_repair',gtbest['index'])]
        for kind,index in cases:
            r=records[index];e=entries[index]
            im=np.array(Image.open(Path('/data6/chensq/datasets')/e['query']['image']).convert('RGB').resize((640,640)))
            with np.load(out/'cache'/f'{shots}shot_{index:03d}.npz') as cache:
                gt=cache['target']>0
                fig,axes=plt.subplots(2,4,figsize=(16,8))
                axes=axes.ravel();axes[0].imshow(im);axes[0].contour(gt,levels=[.5],colors=['cyan'],linewidths=1)
                axes[0].set_title(f'Query + GT (cyan), class {e["class_id"]}')
                names=['B0','S1','S2','G2','G2shuffle','H2']
                for ax,name in zip(axes[1:7],names):
                    pred=F.interpolate(torch.from_numpy(cache[name]),(640,640),mode='bilinear',align_corners=False)[0,0].numpy()>0
                    ax.imshow(im);overlay=np.zeros((640,640,4));overlay[pred]=[1,.2,.1,.4];ax.imshow(overlay)
                    ax.contour(gt,levels=[.5],colors=['cyan'],linewidths=.6)
                    pts=r['rows'][name].get('points',{})
                    for (x,y),lab in zip(pts.get('coords',[]),pts.get('labels',[])):
                        ax.scatter(x*640/1024,y*640/1024,c='lime' if lab else 'magenta',s=36,marker='+' if lab else 'x',linewidths=1.5)
                    ax.set_title(f"{name}: FG IoU {100*episode_iou(r['rows'][name]['counts']):.2f}%")
                ax=axes[7]
                if 'GT2' in cache:
                    pred=F.interpolate(torch.from_numpy(cache['GT2']),(640,640),mode='bilinear',align_corners=False)[0,0].numpy()>0
                    ax.imshow(im);overlay=np.zeros((640,640,4));overlay[pred]=[1,.2,.1,.4];ax.imshow(overlay);ax.contour(gt,levels=[.5],colors=['cyan'],linewidths=.6)
                    d=r['diagnostics']['GT2']
                    for (x,y),lab in zip(d['points']['coords'],d['points']['labels']):
                        ax.scatter(x*640/1024,y*640/1024,c='lime' if lab else 'magenta',s=36,marker='+' if lab else 'x')
                    ax.set_title(f"GT points ORACLE: {100*episode_iou(d['counts']):.2f}%")
                elif 'gradient_response' in cache:
                    ax.imshow(cache['gradient_response'],cmap='magma');ax.set_title('Gradient spatial L2 response')
                for ax in axes:ax.axis('off')
                fig.suptitle(f'{shots}-shot / {r["episode_id"]} / {kind}; predictions orange, GT cyan',fontsize=12)
                fig.tight_layout();path=figure_dir/f'{shots}shot_{kind}_{index:03d}.png';fig.savefig(path,dpi=120);plt.close(fig)
                selected.append(dict(shots=shots,index=index,kind=kind,episode_id=e['episode_id'],query=e['query']['image'],path=str(path)))
    with (out/'figure_index.json').open('x') as f:json.dump(selected,f,indent=2)
    print(json.dumps(selected,indent=2))


if __name__=='__main__':main()
