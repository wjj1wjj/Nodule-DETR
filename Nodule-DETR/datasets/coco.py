# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""COCO box annotations with an explicit category-ID / class-index mapping."""
from pathlib import Path
import torch
from torchvision.datasets import CocoDetection as TorchCocoDetection
from . import transforms as T

def make_coco_transforms(split):
    normalize = T.Compose([T.ToTensor(), T.Normalize(
        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    if split == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(0.5),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([T.RandomResize([400, 500, 600]),
                           T.RandomSizeCrop(384, 600),
                           T.RandomResize(scales, max_size=1333)])), normalize])
    return T.Compose([T.RandomResize([800], max_size=1333), normalize])

class CocoDetection(TorchCocoDetection):
    def __init__(self, root, split, categories=None):
        if split not in {"train", "val", "test"}:
            raise ValueError("Dataset split must be train, val, or test.")
        root = Path(root)
        super().__init__(str(root / f'{split}2017'),
                         str(root / 'annotations' / f'instances_{split}2017.json'))
        own_categories = sorted(self.coco.cats.values(), key=lambda c: c['id'])
        self.categories = categories if categories is not None else own_categories
        expected = {c['id']: c['name'] for c in self.categories}
        actual = {c['id']: c['name'] for c in own_categories}
        if actual != expected:
            raise ValueError('Category IDs and names must agree across dataset splits/checkpoint.')
        self.category_to_label = {c['id']: i for i, c in enumerate(self.categories)}
        self.label_to_category = {i: c['id'] for i, c in enumerate(self.categories)}
        self._transforms = make_coco_transforms(split)

    def __getitem__(self, index):
        image, annotations = super().__getitem__(index)
        w, h = image.size
        annotations = [a for a in annotations if not a.get('iscrowd', 0)]
        boxes = torch.as_tensor([a['bbox'] for a in annotations], dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(0, w)
        boxes[:, 1::2].clamp_(0, h)
        keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        labels = torch.tensor([self.category_to_label[a['category_id']] for a in annotations], dtype=torch.int64)
        area = torch.tensor([a.get('area', a['bbox'][2] * a['bbox'][3]) for a in annotations], dtype=torch.float32)
        target = {'boxes': boxes[keep], 'labels': labels[keep], 'area': area[keep],
                  'iscrowd': torch.zeros(len(annotations), dtype=torch.int64)[keep],
                  'image_id': torch.tensor([self.ids[index]]),
                  'orig_size': torch.tensor([h, w]), 'size': torch.tensor([h, w])}
        return self._transforms(image, target)
