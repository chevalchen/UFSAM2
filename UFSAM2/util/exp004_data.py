"""Fixed, audited identity replay; never sample novel-class queries."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_SHA256 = '2e48ef99362a18d9e33fc00ba7e1c9bbc4ce0fa6b2e5b47e87a1ac8fd8acae40'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def verified_file(item):
    path = Path(item['path'])
    if not path.is_absolute():
        path = ROOT / path
    if sha256(path) != item['sha256']:
        raise ValueError(f'Input SHA256 mismatch: {path}')
    return path


def validate_manifest(manifest, exclusions):
    if (manifest['experiment_id'], manifest['dataset'], manifest['fold'], manifest['shots'], manifest['source_split']) != ('EXP-004', 'coco', 0, 5, 'train'):
        raise ValueError('Wrong data protocol')
    if set(manifest['partitions']) != {'train', 'calibration', 'validation'}:
        raise ValueError('Wrong partitions')
    images, ids = {}, set()
    for part, count in [('train', 1200), ('calibration', 600), ('validation', 600)]:
        entries = manifest['partitions'][part]
        if len(entries) != count:
            raise ValueError('Wrong partition count')
        images[part] = set()
        classes = {}
        for e in entries:
            c = e['class_id']
            if c not in range(80) or c % 4 == 0:
                raise ValueError('Novel or invalid class')
            classes[c] = classes.get(c, 0) + 1
            names = [e['query_id']] + e['support_ids']
            if len(names) != 6 or len(set(names)) != 6:
                raise ValueError('Query/support identities must be distinct')
            if any(not n.startswith('train2014/COCO_train2014_') or '..' in n or not n.endswith('.jpg') for n in names):
                raise ValueError('Invalid image identity')
            if e['episode_id'] in ids:
                raise ValueError('Duplicate episode')
            ids.add(e['episode_id'])
            images[part].update(names)
        if len(classes) != 60 or set(classes.values()) != {count // 60}:
            raise ValueError('Unbalanced class coverage')
        if part != 'train' and images[part] & set(exclusions['image_ids']):
            raise ValueError('Previously observed validation image leakage')
    for a, b in [('train', 'calibration'), ('train', 'validation'), ('calibration', 'validation')]:
        if images[a] & images[b]:
            raise ValueError(f'Cross-partition identity leakage: {a}/{b}')
    return {p: len(v) for p, v in images.items()}


def load_contract(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    if sha256(path) != CONTRACT_SHA256:
        raise ValueError('Frozen scientific contract changed; create an explicitly reviewed revision')
    contract = json.loads(path.read_text())
    if contract['experiment_id'] != 'EXP-004' or contract['schema_version'] != 1:
        raise ValueError('Wrong contract')
    manifest = json.loads(verified_file(contract['manifest']).read_text())
    exclusions = json.loads(verified_file(contract['exclusions']).read_text())
    if manifest['exclusions_sha256'] != contract['exclusions']['sha256']:
        raise ValueError('Exclusion provenance mismatch')
    validate_manifest(manifest, exclusions)
    metadata = Path(contract['data_root']) / 'splits/trn/fold0.pkl'
    if sha256(metadata) != manifest['metadata_sha256']:
        raise ValueError('COCO split metadata changed')
    return contract, manifest


class FixedEpisodes:
    def __init__(self, manifest, partition, data_root):
        self.entries = manifest['partitions'][partition]
        self.root = Path(data_root)
        self.transform = transforms.Compose([transforms.Resize((640, 640)), transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        e = self.entries[index]
        images, masks = [], []
        for name in e['support_ids'] + [e['query_id']]:
            with Image.open(self.root / name) as image:
                images.append(self.transform(image.convert('RGB')))
            with Image.open(self.root / 'annotations' / str(Path(name).with_suffix('.png'))) as annotation:
                mask = torch.from_numpy((np.array(annotation) == e['class_id'] + 1).astype(np.float32))
            masks.append(F.interpolate(mask[None, None], (640, 640), mode='nearest')[0, 0])
        return torch.stack(images)[None], torch.stack(masks[:5]), masks[5], e
