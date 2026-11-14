"""Deterministic replay of four COCO training episodes; no dataset protocol changes."""
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from datasets.coco import build


def batch_to_sequence(batch):
    return (torch.cat([batch['support_imgs'], batch['query_img'].unsqueeze(1)], 1),
            torch.cat([batch['support_masks'], batch['query_mask'].unsqueeze(1)], 1))


def fixed_loader(root, shots, batch_size, steps):
    dataset = build('train', SimpleNamespace(data_root=root, fold=0, shots=shots))
    original_load = dataset.load_frame
    records = []

    def traced_load():
        values = original_load()
        records.append(dict(query=values[4], supports=values[5], class_id=int(values[6])))
        return values

    dataset.load_frame = traced_load

    class Replay(Dataset):
        def __len__(self):
            return max(4, steps) * batch_size

        def __getitem__(self, index):
            state = np.random.get_state()
            try:
                np.random.seed(123 + index % (4 * batch_size))
                return dataset[index]
            finally:
                np.random.set_state(state)

    replay = Replay()
    hashes = []
    for index in range(4 * batch_size):
        batch = replay[index]
        digest = hashlib.sha256()
        for key in ('query_img', 'query_mask', 'support_imgs', 'support_masks'):
            digest.update(batch[key].contiguous().numpy().tobytes())
        hashes.append(digest.hexdigest())
    dataset.load_frame = original_load
    split = Path(root) / 'COCO2014/splits/trn/fold0.pkl'
    manifest = dict(dataset='COCO2014', fold=0, split='trn', episodes=records,
                    tensor_sha256=hashes, split_sha256=hashlib.sha256(split.read_bytes()).hexdigest(),
                    transform=str(dataset.transform), input_size=640, crop=None,
                    replay='four episodes, seeds 123..126 (batch=1), cyclic',
                    workers=0, pin_memory=False)
    return DataLoader(replay, batch_size=batch_size, num_workers=0), manifest
