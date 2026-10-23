"""Explicit frame/instance replay for COCO and cropped part segmentation."""
import json,pickle,re
from pathlib import Path
from functools import lru_cache
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torchvision import transforms
import pycocotools.mask as mask_utils
from util.exp004_data import sha256

ROOT=Path(__file__).resolve().parents[2]
DATA=Path('/data6/chensq/datasets')

@lru_cache(maxsize=1)
def paco_annotations():
    return pickle.load(open(DATA/'PACO-Part/paco/paco_part_val.pkl','rb'))['img2anno']


def mask_decode(seg,height,width):
    if isinstance(seg,list):seg=mask_utils.merge(mask_utils.frPyObjects(seg,height,width))
    if isinstance(seg,dict) and isinstance(seg.get('counts'),list):seg=mask_utils.frPyObjects(seg,height,width)
    out=mask_utils.decode(seg)
    return out.any(-1) if out.ndim==3 else out>0


def frame_data(frame):
    path=DATA/frame['image']
    with Image.open(path) as im:image=im.convert('RGB')
    w,h=image.size
    if frame['dataset']=='coco':
        with Image.open(DATA/frame['annotation']) as im:mask=np.array(im)==frame['class_id']+1
    elif frame['dataset']=='pascal_part':
        annotation=json.loads((DATA/frame['annotation']).read_text())['object'][frame['object_index']]
        masks=[mask_decode(s,h,w) for p in annotation['parts'] if p['name']==frame['part'] for s in p['mask']]
        mask=np.logical_or.reduce(masks)
    else:
        annotations=[a for a in paco_annotations()[frame['image_id']] if a['category_id']==frame['category_id'] and a['obj_ann_id']==frame['object_id']]
        mask=np.logical_or.reduce([mask_decode(a['segmentation'],h,w) for a in annotations])
    if 'crop' in frame:
        x0,y0,x1,y1=frame['crop'];image=Image.fromarray(np.asarray(image)[y0:y1,x0:x1]);mask=mask[y0:y1,x0:x1]
    if not mask.size or not mask.any():raise ValueError(f'Empty foreground/crop: {frame}')
    return image,torch.from_numpy(mask.astype(np.float32))


class Episodes:
    def __init__(self,manifest,partition):
        self.entries=manifest['partitions'][partition]
        self.transforms={size:transforms.Compose([transforms.Resize((size,size)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])]) for size in (518,640)}
    def __len__(self):return len(self.entries)
    def __getitem__(self,index):
        e=self.entries[index];size=518 if e['dataset']=='pascal_part' else 640
        images=[];masks=[]
        for frame in e['supports']+[e['query']]:
            image,mask=frame_data(frame);images.append(self.transforms[size](image));masks.append(F.interpolate(mask[None,None],(size,size),mode='nearest')[0,0])
        return torch.stack(images)[None],torch.stack(masks[:5]),masks[5],e


def validate(manifest):
    sets={};episodes=set()
    for part,entries in manifest['partitions'].items():
        sets[part]=set()
        for e in entries:
            if e['episode_id'] in episodes:raise ValueError('Duplicate episode')
            episodes.add(e['episode_id'])
            ids=[f['identity'] for f in e['supports']+[e['query']]]
            if len(ids)!=6 or len(set(ids))!=6:raise ValueError('Duplicated support/query identity')
            sets[part].update(ids)
            if e['dataset']=='coco' and (e['class_id']%4==0 or not 0<=e['class_id']<80):raise ValueError('Novel COCO class accessed')
    for a,b in [('train','calibration'),('train','coco'),('calibration','coco'),('train','paco_part'),('calibration','paco_part')]:
        if sets[a]&sets[b]:raise ValueError(f'Image leakage {a}/{b}')
    excluded=set(manifest['prior_validation_identities'])
    if (sets['calibration']|sets['coco'])&excluded:raise ValueError('Prior validation reused')
    for part,count in [('train',1200),('calibration',600),('coco',600)]:
        if len(manifest['partitions'][part])!=count:raise ValueError('Wrong COCO count')
    return {k:len(v) for k,v in sets.items()}


def load_inputs(config_path):
    c=json.loads(Path(config_path).read_text());p=ROOT/c['manifest']['path']
    if sha256(p)!=c['manifest']['sha256']:raise ValueError('Manifest changed')
    m=json.loads(p.read_text());validate(m)
    for item in m['metadata']:
        if sha256(item['path'])!=item['sha256']:raise ValueError('Dataset metadata changed')
    return c,m
