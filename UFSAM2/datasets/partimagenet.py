"""Full-image PartImageNet episodic evaluation from the official COCO-style part JSON."""
from collections import defaultdict
from pathlib import Path
import random

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class PartImageNet(Dataset):
    def __init__(self, annotation, image_root, shots, seed):
        self.coco = COCO(annotation)
        self.image_root = Path(image_root)
        self.shots = shots
        self.seed = seed
        self.transform = transforms.Compose([
            transforms.Resize((518, 518)), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        by_category = defaultdict(set)
        for ann in self.coco.anns.values():
            if not ann.get("iscrowd", 0) and ann.get("area", 0) > 0:
                by_category[ann["category_id"]].add(ann["image_id"])
        self.pools = {cat: sorted(images) for cat, images in sorted(by_category.items())
                      if len(images) > shots}
        if not self.pools:
            raise ValueError("No part category has enough distinct support and query images")
        self.class_ids = list(self.pools)
        self.nclass = max(self.coco.cats) + 1
        self.episodes = [(cat, image) for cat, images in self.pools.items() for image in images]

    def __len__(self):
        return len(self.episodes)

    def _read(self, category, image_id):
        info = self.coco.imgs[image_id]
        with Image.open(self.image_root / info["file_name"]) as image:
            rgb = self.transform(image.convert("RGB"))
        anns = self.coco.getAnnIds(imgIds=[image_id], catIds=[category], iscrowd=False)
        mask = np.zeros((info["height"], info["width"]), dtype=np.uint8)
        for ann in self.coco.loadAnns(anns):
            mask |= self.coco.annToMask(ann)
        mask = Image.fromarray(mask).resize((518, 518), Image.Resampling.NEAREST)
        return rgb, torch.from_numpy(np.array(mask)).float()

    def __getitem__(self, index):
        category, query = self.episodes[index]
        rng = random.Random(f"{self.seed}:{category}:{query}")
        supports = rng.sample([i for i in self.pools[category] if i != query], self.shots)
        image, mask = self._read(category, query)
        imgs, masks = zip(*(self._read(category, s) for s in supports))
        return dict(query_img=image, query_mask=mask, query_name=str(query),
                    support_imgs=torch.stack(imgs), support_masks=torch.stack(masks),
                    support_names=list(map(str, supports)), class_id=torch.tensor(category))
