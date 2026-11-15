"""Prepare missing SANSA metadata without modifying ADE20K images or annotations."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch


def audit_pair(paths):
    image_path, mask_path = paths
    with Image.open(image_path) as image:
        image.load()
        size = image.size
    with Image.open(mask_path) as mask:
        mask.load()
        if size != mask.size:
            raise ValueError(f'Size mismatch: {image_path}')
        labels = np.unique(np.asarray(mask.convert('L'))).tolist()
    if any(x < 0 or x > 150 for x in labels):
        raise ValueError(f'Unexpected labels: {mask_path}: {labels}')
    return image_path.stem, [x for x in labels if x not in (0, 255)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--report', required=True)
    args = parser.parse_args()
    root = Path(args.root)
    utility = root / 'utils'
    utility.mkdir(exist_ok=True)
    rows = [line.split('\t') for line in (root/'objectInfo150.txt').read_text().splitlines()[1:]]
    assert [int(row[0]) for row in rows] == list(range(1, 151))
    classes = [row[-1] for row in rows]
    path = utility/'ade20k_classes.json'
    if path.exists():
        assert json.loads(path.read_text()) == classes
    else:
        with path.open('x') as stream:
            json.dump(classes, stream, indent=2)
    report = {'root': str(root), 'classes': 150, 'splits': {}, 'sha256': {}}
    for split, prefix, expected in [('training', 'train', 20210), ('validation', 'val', 2000)]:
        images = sorted((root/'images'/split).glob('*.jpg'))
        masks = sorted((root/'annotations'/split).glob('*.png'))
        assert len(images) == len(masks) == expected
        assert [x.stem for x in images] == [x.stem for x in masks]
        metadata, ids = defaultdict(list), []
        with ThreadPoolExecutor(max_workers=4) as pool:
            for index, (name, labels) in enumerate(pool.map(audit_pair, zip(images, masks))):
                if labels:
                    ids.append(name)
                for label in labels:
                    metadata[label].append(name)
                if (index+1) % 2000 == 0:
                    print(split, index+1, flush=True)
        cache = utility/f'{prefix}_ade20k_icl.pth'
        if cache.exists():
            old_metadata, old_ids = torch.load(cache, weights_only=False)
            assert old_metadata == metadata and old_ids == ids
        else:
            with cache.open('xb') as stream:
                torch.save([metadata, ids], stream)
        report['splits'][split] = dict(images=len(images), masks=len(masks), valid_images=len(ids),
                                      class_counts={str(k):len(v) for k,v in sorted(metadata.items())})
        print(split, report['splits'][split]['valid_images'], 'valid; minimum class count', min(map(len,metadata.values())), flush=True)
    for path in [root/'objectInfo150.txt', *sorted(utility.iterdir())]:
        report['sha256'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    with open(args.report, 'x') as stream:
        json.dump(report, stream, indent=2)
    print('ADE20K preparation and full decode audit passed', flush=True)


if __name__ == '__main__':
    main()
