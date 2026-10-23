import sys,json,concurrent.futures,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'UFSAM2'))
import torch
from torch.nn import functional as F
from util.exp005_data import ROOT,frame_data,paco_annotations,validate
from util.exp004_data import sha256

def check(frame):
 try:
  image,mask=frame_data(frame);size=518 if frame['dataset']=='pascal_part' else 640
  resized=F.interpolate(mask[None,None],(size,size),mode='nearest')
  if not F.interpolate(resized,(64,64),mode='nearest').any():return dict(frame=frame,error='Empty foreground at encoder resolution')
  return None
 except Exception as e:return dict(frame=frame,error=repr(e))

if __name__=='__main__':
 torch.set_num_threads(1);p=ROOT/'experiments/EXP-005/episodes.json';m=json.loads(p.read_text());validate(m);paco_annotations()
 frames={json.dumps(f,sort_keys=True):f for entries in m['partitions'].values() for e in entries for f in e['supports']+[e['query']]}
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:bad=[r for r in executor.map(check,frames.values()) if r]
 report=dict(manifest_sha256=sha256(p),unique_frames=len(frames),failures=bad)
 with open(sys.argv[1],'x') as f:json.dump(report,f,indent=2)
 print(json.dumps(dict(unique_frames=len(frames),failures=len(bad))),flush=True)
 if bad:raise SystemExit(1)
