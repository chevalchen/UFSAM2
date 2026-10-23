import sys,json,pickle,random,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'UFSAM2'))
from util.exp005_data import DATA,ROOT,validate
from util.exp004_data import sha256

rng=random.Random(5)
previous=json.loads((ROOT/'experiments/EXP-004/episodes/prior_validation_exclusions.json').read_text())['image_ids']
old=json.loads((ROOT/'experiments/EXP-004/episodes/coco_fold0_5shot_fresh_seed0.json').read_text())
previous=set(previous)|{n for e in old['partitions']['validation'] for n in [e['query_id']]+e['support_ids']}
def coco_id(name):return 'coco:'+str(int(Path(name).stem.split('_')[-1]))
excluded={coco_id(n) for n in previous}
metadata=DATA/'COCO2014/splits/trn/fold0.pkl';meta=pickle.load(open(metadata,'rb'))
classes=[c for c in range(80) if c%4]
assignment={n:'train' for n in previous}
for c in sorted(classes,key=lambda c:len(set(meta[c])-previous)):
    names=sorted(set(meta[c]),key=lambda n:hashlib.sha256(('EXP005:'+n).encode()).hexdigest())
    for p in ['calibration','coco']:
        need=max(0,6-sum(assignment.get(n)==p for n in names))
        available=[n for n in names if n not in assignment]
        if len(available)<need:raise ValueError(f'Insufficient fresh images {c}/{p}')
        assignment.update({n:p for n in available[:need]})
for n in sorted({n for c in classes for n in meta[c]}):
    assignment.setdefault(n,['train','calibration','coco'][int(hashlib.sha256(('EXP005:'+n).encode()).hexdigest(),16)%3])
def coco_frame(n,c):return dict(dataset='coco',class_id=c,image='COCO2014/'+n,annotation='COCO2014/annotations/'+str(Path(n).with_suffix('.png')),identity=coco_id(n))
parts={};audit={}
for p,num in [('train',20),('calibration',10),('coco',10)]:
    entries=[]
    for c in classes:
        pool=sorted(n for n in set(meta[c]) if assignment[n]==p)
        if len(pool)<6:raise ValueError(f'Insufficient pool {p}/{c}')
        for i in range(num):
            names=rng.sample(pool,6);entries.append(dict(dataset='coco',episode_id=f'{p}-{c}-{i}',class_id=c,group=0,query=coco_frame(names[0],c),supports=[coco_frame(n,c) for n in names[1:]]))
    rng.shuffle(entries);parts[p]=entries
blocked={f['identity'] for p in ('train','calibration') for e in parts[p] for f in e['supports']+[e['query']]}
metadata_paths=[metadata]
# Preserve original Pascal evaluation split and instance crop convention.
pascal=DATA/'Pascal-Part/VOCdevkit/VOC2010';source=pascal/'all_obj_part_to_image.json';metadata_paths.append(source)
js=json.loads(source.read_text());pools={};exclusions=[]
for group,cat in enumerate(['animals','indoor','person','vehicles']):
    for obj,info in js[cat]['object'].items():
        for part,lists in info['part'].items():
            key=obj+'+'+part
            if not lists['train'] or not lists['val'] or key=='aeroplane+TAIL':continue
            frames=[]
            for name in sorted(set(lists['val'])):
                annopath=pascal/'Annotations_Part_json_merged_part_classes'/f'{name}.json'
                anno=json.loads(annopath.read_text())
                options=[]
                for index,o in enumerate(anno['object']):
                    if o['name']!=obj or not any(p['name']==part and p['mask'] for p in o['parts']):continue
                    box=[int(o['bndbox'][k]) for k in ['xmin','ymin','xmax','ymax']]
                    if box[2]<=box[0] or box[3]<=box[1]:continue
                    options.append(dict(dataset='pascal_part',image=str((pascal/'JPEGImages'/f'{name}.jpg').relative_to(DATA)),annotation=str(annopath.relative_to(DATA)),object_index=index,part=part,crop=box,identity='pascal:'+name))
                if options:frames.append(rng.choice(options))
            if len(frames)>=6:pools[(group,key)]=frames
            else:exclusions.append(dict(group=group,category=key,images=len(frames)))
def make_parts(name,pools,exclusions):
    entries=[]
    for group in range(4):
        keys=sorted(k for k in pools if k[0]==group)
        if not keys:continue
        for i in range(250):
            key=keys[i%len(keys)];frames=rng.sample(pools[key],6)
            entries.append(dict(dataset=name,episode_id=f'{name}-{group}-{i}',class_id=str(key[0])+':'+str(key[1]),group=group,query=frames[0],supports=frames[1:]))
    parts[name]=entries;audit[name]=dict(episodes=len(entries),eligible_classes=len(pools),excluded_classes=exclusions,groups={str(g):sum(k[0]==g for k in pools) for g in range(4)})
make_parts('pascal_part',pools,exclusions)
# PACO categories retain the original train-dictionary order used to define folds.
pacodir=DATA/'PACO-Part/paco';trainpath=pacodir/'paco_part_train.pkl';valpath=pacodir/'paco_part_val.pkl';metadata_paths += [trainpath,valpath]
train=pickle.load(open(trainpath,'rb'));val=pickle.load(open(valpath,'rb'));pools={};exclusions=[]
for index,c in enumerate(train['cid2img']):
    frames=[];seen=set();group=index%4
    for item in val['cid2img'].get(c,[]):
        image_id,name=next(iter(item.items()));identity='coco:'+str(int(image_id))
        if identity in blocked or identity in seen:continue
        seen.add(identity);name='/'.join(name.split('/')[-2:])
        objects={}
        for a in val['img2anno'][image_id]:
            if a['category_id']==c:objects.setdefault(a['obj_ann_id'],a['obj_bbox'])
        candidates=[]
        for obj,b in objects.items():
            box=[int(b[0]),int(b[1]),int(b[0]+b[2]),int(b[1]+b[3])]
            if box[2]>box[0] and box[3]>box[1]:candidates.append(dict(dataset='paco_part',image='PACO-Part/coco/'+name,image_id=int(image_id),category_id=int(c),object_id=int(obj),crop=box,identity=identity))
        if candidates:frames.append(rng.choice(candidates))
    if len(frames)>=6:pools[(group,int(c))]=frames
    else:exclusions.append(dict(group=group,category=int(c),images=len(frames)))
make_parts('paco_part',pools,exclusions)
manifest=dict(experiment='EXP-005',seed=5,prior_validation_identities=sorted(excluded),partitions=parts,audit=audit,metadata=[dict(path=str(p),sha256=sha256(p)) for p in metadata_paths])
print(validate(manifest));print(audit)
with (ROOT/'experiments/EXP-005/episodes.json').open('x') as f:json.dump(manifest,f,indent=2)
