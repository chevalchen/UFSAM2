"""Preflight input integrity only: foreground availability, never model scores."""
import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image


def check(task):
    root, name, classes = task
    path = root / 'annotations' / str(Path(name).with_suffix('.png'))
    contents = path.read_bytes()
    with Image.open(path) as image:
        mask = np.array(image)
    # Original->640 nearest, then ->64 nearest equals floor(i*original_size/64).
    small = mask[(np.arange(64)*mask.shape[0]//64)[:,None], (np.arange(64)*mask.shape[1]//64)[None,:]]
    empty = [c for c in classes if not np.any(small == c+1)]
    return dict(image_id=name, annotation_sha256=hashlib.sha256(contents).hexdigest(),
                empty_classes=empty)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',required=True)
    p.add_argument('--data-root',required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    manifest=json.loads(Path(args.manifest).read_text())
    items={}
    supports=set()
    for part,entries in manifest['partitions'].items():
        for e in entries:
            for name in e['support_ids']+[e['query_id']]:
                items.setdefault(name,set()).add(e['class_id'])
            supports.update((n,e['class_id']) for n in e['support_ids'])
    with ThreadPoolExecutor(max_workers=8) as executor:
        results=list(executor.map(check,[(Path(args.data_root),n,sorted(c)) for n,c in sorted(items.items())]))
    bad=[dict(image_id=r['image_id'],class_id=c) for r in results for c in r['empty_classes'] if (r['image_id'],c) in supports]
    report=dict(manifest_sha256=hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
                unique_images=len(items),empty_support_cases=bad,annotations=results)
    with open(args.output,'x') as f:json.dump(report,f,indent=2)
    print(json.dumps(dict(unique_images=len(items),empty_support_cases=bad)))

if __name__=='__main__':main()
