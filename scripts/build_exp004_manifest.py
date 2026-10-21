"""Deterministic identity-disjoint base-only episodes; no segmentation scores used."""
import argparse
import hashlib
import json
import pickle
import random
from collections import Counter
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def generate(metadata, exclusions):
    classes = [c for c in range(80) if c % 4 != 0]
    def partition(name):
        if name in exclusions:
            return 'train'
        bucket = int(hashlib.sha256(('EXP-004:0:' + name).encode()).hexdigest(), 16) % 3
        return ['train', 'calibration', 'validation'][bucket]
    rng = random.Random(0)
    parts, audit = {}, {}
    for part, per_class in [('train', 20), ('calibration', 10), ('validation', 10)]:
        episodes, sizes = [], {}
        for c in classes:
            pool = sorted(n for n in set(metadata[c]) if partition(n) == part)
            if len(pool) < 6:
                raise ValueError(f'Insufficient pool: {part}/{c}: {len(pool)}')
            sizes[str(c)] = len(pool)
            for index in range(per_class):
                names = rng.sample(pool, 6)
                episodes.append(dict(episode_id=f'{part}-{c:02d}-{index:02d}', class_id=c,
                                     query_id=names[0], support_ids=names[1:]))
        rng.shuffle(episodes)
        parts[part] = episodes
        audit[part] = dict(pool_sizes=sizes, class_counts=dict(sorted(Counter(e['class_id'] for e in episodes).items())),
                           unique_images=len({n for e in episodes for n in [e['query_id']] + e['support_ids']}))
    return dict(schema_version=1, experiment_id='EXP-004', dataset='coco', fold=0, shots=5,
                source_split='train', generation_seed=0, partitions=parts, audit=audit)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--metadata', required=True)
    p.add_argument('--exclusions', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    exc = json.loads(Path(a.exclusions).read_text())
    with open(a.metadata, 'rb') as f:
        data = generate(pickle.load(f), set(exc['image_ids']))
    data['metadata_sha256'] = sha(a.metadata)
    data['exclusions_sha256'] = sha(a.exclusions)
    with open(a.output, 'x') as f:
        json.dump(data, f, indent=2)
        f.write('\n')

if __name__ == '__main__':
    main()
