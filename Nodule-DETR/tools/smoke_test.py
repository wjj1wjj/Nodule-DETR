"""CPU checks of the full paper model; no data or weights are downloaded."""
import json
from pathlib import Path
import sys
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.ops import generalized_box_iou as reference_giou
from main import get_args_parser
from models import build_nodule_detr
from datasets import CocoDetection
from datasets.coco import make_coco_transforms
from datasets import transforms as T
from util import box_ops
from util.misc import collate_fn
from engine import train_one_epoch, evaluate


def run():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    args = get_args_parser().parse_args(['--device', 'cpu', '--num_workers', '0'])
    model, criterion, processors = build_nodule_detr(args)
    assert len(model.transformer.encoder.layers) == len(model.transformer.decoder.layers) == 6
    assert sum(1 for m in model.modules() if m.__class__.__name__ == 'MSFCA') == 16
    assert model.backbone[1].temperature == model.backbone[1].temperature_w == 20
    assert criterion.weight_dict['loss_ce'] == criterion.weight_dict['tgt_loss_ce'] == 2
    # Standard GIoU must include corners outside the image, without clamping.
    boxes = torch.tensor([[-0.2, 0.1, 0.5, 0.8], [0.6, 0.6, 1.2, 1.3]])
    assert torch.allclose(box_ops.generalized_box_iou(boxes, boxes), reference_giou(boxes, boxes), atol=2e-5)
    categories = [{'id': 3, 'name': 'benign'}, {'id': 17, 'name': 'malignant'}]
    with tempfile.TemporaryDirectory(prefix='nodule-smoke-') as directory:
        root = Path(directory)
        (root / 'annotations').mkdir()
        for split in ['train', 'val', 'test']:
            (root / f'{split}2017').mkdir()
            images, annotations = [], []
            for i in range(2):
                Image.new('RGB', (128, 96), (40 + i * 20, 70, 90)).save(root / f'{split}2017' / f'{i}.png')
                images.append({'id': i, 'file_name': f'{i}.png', 'height': 96, 'width': 128})
                annotations.append({'id': i+1, 'image_id': i, 'category_id': categories[i]['id'],
                                    'bbox': [30, 20, 40, 35], 'area': 1400, 'iscrowd': 0})
            (root / 'annotations' / f'instances_{split}2017.json').write_text(json.dumps(
                {'images': images, 'annotations': annotations, 'categories': categories}))
        train = CocoDetection(root, 'train')
        val = CocoDetection(root, 'val', train.categories)
        # Verify paper evaluation resize independently; keep full-model CPU checks small.
        tensor, _ = make_coco_transforms('val')(Image.new('RGB', (128, 96)), None)
        assert tuple(tensor.shape[-2:]) == (800, 1066)
        small_transform = T.Compose([T.ToTensor(), T.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        train._transforms = val._transforms = small_transform
        assert [train[i][1]['labels'].item() for i in range(2)] == [0, 1]
        loader = DataLoader(train, batch_size=2, collate_fn=collate_fn)
        groups = [
            {'params': [p for n,p in model.named_parameters() if p.requires_grad and 'backbone' not in n]},
            {'params': [p for n,p in model.named_parameters() if p.requires_grad and 'backbone' in n], 'lr': args.lr_backbone}]
        optimizer = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)
        stats = train_one_epoch(model, criterion, loader, optimizer, torch.device('cpu'), 0, args)
        assert stats['loss'] > 0
        assert model.hff.scDown1.cv1.conv.weight.grad is not None
        assert model.hff.scDown2.cv1.conv.weight.grad is not None
        assert model.hff.scDown3.cv1.conv.weight.grad is None  # Historical unused parameter retained.
        val_loader = DataLoader(val, batch_size=2, collate_fn=collate_fn)
        stats = evaluate(model, processors['bbox'], val_loader, torch.device('cpu'))
        assert len(stats['coco_bbox']) == 12
        model.train()
        samples, targets = next(iter(loader))
        targets[0]['labels'] = torch.empty(0, dtype=torch.int64)
        targets[0]['boxes'] = torch.empty(0, 4)
        for empty_all in [False, True]:
            if empty_all:
                targets[1]['labels'] = torch.empty(0, dtype=torch.int64)
                targets[1]['boxes'] = torch.empty(0, 4)
            optimizer.zero_grad(set_to_none=True)
            output, meta = model(samples, dn_args=(targets, 5, .2, .4, 0))
            loss_dict = criterion(output, targets, meta)
            loss = sum(v * criterion.weight_dict[k] for k, v in loss_dict.items() if k in criterion.weight_dict)
            assert torch.isfinite(loss)
            loss.backward()
        model.eval()
        with torch.no_grad():
            output, meta = model(samples, dn_args=0)
        assert meta is None
        assert output['pred_logits'].shape == (2, 300, 2)
        assert output['pred_boxes'].shape == (2, 300, 4)
    print('PASS: full 6/6 model, MSFCA/HFF/MSDA, loss/backward, empty targets, sparse category IDs, COCO evaluation, inference without DN.')


if __name__ == '__main__':
    run()
