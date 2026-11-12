"""Reproduce (or verify) the fixed EXP-006 nested-shot development selection."""
import argparse
import hashlib
import json
from pathlib import Path
import random


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--verify', action='store_true')
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = Path('experiments/EXP-005/episodes.json')
    m = json.loads((root/source).read_text())
    groups = {}
    for e in m['partitions']['coco']:
        groups.setdefault(e['class_id'], []).append(e)
    rng = random.Random(6006); entries = []
    for c in sorted(groups):
        for j, source_entry in enumerate(rng.sample(groups[c], 2)):
            e = dict(source_entry)
            e.update(source_episode_id=e['episode_id'], episode_id=f'exp006-dev-c{c:02d}-{j}',
                     gt_diagnostic=c in sorted(groups)[::5])
            assert c % 4 != 0 and len({f['identity'] for f in e['supports']+[e['query']]}) == 6
            entries.append(e)
    manifest = dict(experiment='EXP-006', version=1, seed=6006,
        evidence='DEVELOPMENT_REUSED_EXP005_BASE', fold=0, classes='base (class_id % 4 != 0)',
        shots=[1,5], one_shot_support_index=0,
        source=dict(path=str(source),sha256=hashlib.sha256((root/source).read_bytes()).hexdigest(),partition='coco'),
        metadata=m['metadata'], entries=entries)
    serialized = json.dumps(manifest,indent=2)+'\n'
    destination = root/'experiments/EXP-006/episodes_v1.json'
    if a.verify:
        assert serialized == destination.read_text(), 'Fixed manifest does not reproduce'
    else:
        with destination.open('x') as f:
            f.write(serialized)
    prior = set(m['prior_validation_identities'])
    identities = {f['identity'] for e in entries for f in e['supports']+[e['query']]}
    print(json.dumps(dict(episodes=len(entries),query_images=len({e['query']['identity'] for e in entries}),
        all_images=len(identities), prior_exp001_to_004_validation_image_overlap=len(identities&prior),
        reused_exp005_validation_episodes=len(entries),
        sha256=hashlib.sha256(serialized.encode()).hexdigest())))


if __name__ == '__main__':
    main()
