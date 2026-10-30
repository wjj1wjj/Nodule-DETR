# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""Single-GPU training and COCO bounding-box evaluation."""
import math
import torch
from datasets.coco_eval import CocoEvaluator
from util.misc import MetricLogger, SmoothedValue

def train_one_epoch(model, criterion, loader, optimizer, device, epoch, args):
    model.train()
    criterion.train()
    metrics = MetricLogger(delimiter='  ')
    metrics.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    for samples, targets in metrics.log_every(loader, 10, f'Epoch: [{epoch}]'):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs, dn_meta = model(samples, dn_args=(targets, args.scalar,
            args.label_noise_scale, args.box_noise_scale, args.num_patterns))
        losses = criterion(outputs, targets, dn_meta)
        loss = sum(v * criterion.weight_dict[k] for k, v in losses.items() if k in criterion.weight_dict)
        if not math.isfinite(loss.item()):
            raise FloatingPointError(f'Non-finite training loss: {loss.item()}')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
        optimizer.step()
        metrics.update(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])
    return {k: m.global_avg for k, m in metrics.meters.items()}

@torch.no_grad()
def evaluate(model, postprocessor, loader, device):
    if len(loader.dataset) == 0:
        raise ValueError('The evaluation dataset contains no images.')
    model.eval()
    evaluator = CocoEvaluator(loader.dataset.coco, ['bbox'])
    category_map = loader.dataset.label_to_category
    for samples, targets in MetricLogger().log_every(loader, 10, 'Evaluation:'):
        outputs, _ = model(samples.to(device), dn_args=0)
        sizes = torch.stack([t['orig_size'] for t in targets]).to(device)
        predictions = postprocessor(outputs, sizes)
        for prediction in predictions:
            prediction['labels'] = torch.tensor(
                [category_map[i] for i in prediction['labels'].tolist()],
                dtype=torch.int64, device=device)
        evaluator.update({t['image_id'].item(): p for t, p in zip(targets, predictions)})
    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    evaluator.summarize()
    return {'coco_bbox': evaluator.coco_eval['bbox'].stats.tolist()}
