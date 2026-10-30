# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""Nodule-DETR: paper configuration, training, validation and test evaluation."""
import argparse
import json
import random
from pathlib import Path

def get_args_parser():
    p = argparse.ArgumentParser('Nodule-DETR')
    p.add_argument('--coco_path', type=Path, default=Path('data/coco'))
    p.add_argument('--output_dir', type=Path, default=Path('outputs/nodule_detr'))
    p.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--num_classes', type=int, default=2)
    p.add_argument('--batch_size', type=int, default=2)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--lr_backbone', type=float, default=1e-5)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--lr_drop', type=int, default=40)
    p.add_argument('--clip_max_norm', type=float, default=0.1)
    p.add_argument('--enc_layers', type=int, default=6)
    p.add_argument('--dec_layers', type=int, default=6)
    p.add_argument('--hidden_dim', type=int, default=256)
    p.add_argument('--dim_feedforward', type=int, default=2048)
    p.add_argument('--nheads', type=int, default=8)
    p.add_argument('--num_queries', type=int, default=300)
    p.add_argument('--num_feature_levels', type=int, default=4, choices=[4])
    p.add_argument('--enc_n_points', type=int, default=4)
    p.add_argument('--dec_n_points', type=int, default=4)
    p.add_argument('--pe_temperatureH', type=int, default=20)
    p.add_argument('--pe_temperatureW', type=int, default=20)
    p.add_argument('--scalar', type=int, default=5, help='Denoising groups')
    p.add_argument('--label_noise_scale', type=float, default=0.2)
    p.add_argument('--box_noise_scale', type=float, default=0.4)
    p.add_argument('--cls_loss_coef', type=float, default=2)
    p.add_argument('--bbox_loss_coef', type=float, default=5)
    p.add_argument('--giou_loss_coef', type=float, default=2)
    p.add_argument('--set_cost_class', type=float, default=2)
    p.add_argument('--set_cost_bbox', type=float, default=5)
    p.add_argument('--set_cost_giou', type=float, default=2)
    p.add_argument('--focal_alpha', type=float, default=0.25)
    p.add_argument('--resume', type=Path, help='Resume a checkpoint created by this release')
    p.add_argument('--eval', action='store_true')
    p.add_argument('--eval_split', choices=['val', 'test'], default='val')
    p.add_argument('--save_every', type=int, default=50)
    p.set_defaults(backbone='resnet50_msfca', masks=False, dilation=False,
                   position_embedding='sine', dropout=0.0, aux_loss=True,
                   two_stage=False, num_patterns=0, use_dn=True)
    return p

def load_checkpoint(path, device='cpu'):
    import torch
    # Load only checkpoints from a trusted source (PyTorch pickle format).
    return torch.load(path, map_location=device, weights_only=False)

def main(args):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, BatchSampler
    from datasets import CocoDetection
    from models import build_nodule_detr
    from util.misc import collate_fn
    from engine import train_one_epoch, evaluate

    if min(args.batch_size, args.epochs, args.lr_drop, args.save_every, args.num_classes) <= 0:
        raise ValueError('Batch size, epochs, LR step, save interval and class count must be positive.')
    if args.num_workers < 0:
        raise ValueError('--num_workers must be nonnegative.')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Select an available device.')
    if args.eval and args.resume is None:
        raise ValueError('--eval requires --resume with trained weights.')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    checkpoint = load_checkpoint(args.resume) if args.resume else None
    if checkpoint:
        runtime = {'coco_path', 'output_dir', 'device', 'num_workers', 'resume', 'eval', 'eval_split', 'save_every'}
        saved = checkpoint['config']
        if args.eval:
            # Evaluation reconstructs the saved model while retaining runtime options.
            for key, value in saved.items():
                if key not in runtime | {'batch_size'} and hasattr(args, key):
                    setattr(args, key, value)
        else:
            # An explicitly increased training duration can extend a resumed run.
            different = [k for k, v in vars(args).items()
                         if k not in runtime | {'epochs'} and saved.get(k) != v]
            if different:
                raise ValueError('Checkpoint configuration differs; supply its original arguments: ' + ', '.join(different))
    categories = checkpoint['categories'] if checkpoint else None
    train_set = None
    if not args.eval:
        train_set = CocoDetection(args.coco_path, 'train', categories)
        categories = train_set.categories
    eval_set = CocoDetection(args.coco_path, args.eval_split if args.eval else 'val', categories)
    categories = eval_set.categories
    if not len(eval_set):
        raise ValueError('The evaluation split contains no images.')
    if len(categories) != args.num_classes:
        raise ValueError(f'Found {len(categories)} categories; expected --num_classes {args.num_classes}.')
    eval_loader = DataLoader(eval_set, batch_size=args.batch_size, sampler=SequentialSampler(eval_set),
                             num_workers=args.num_workers, collate_fn=collate_fn)
    model, criterion, postprocessors = build_nodule_detr(args)
    model.to(device)
    if checkpoint:
        model.load_state_dict(checkpoint['model'], strict=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    if args.eval:
        (args.output_dir / f'{args.eval_split}_config.json').write_text(json.dumps(config, indent=2))
        stats = evaluate(model, postprocessors['bbox'], eval_loader, device)
        (args.output_dir / f'{args.eval_split}_metrics.json').write_text(json.dumps(stats, indent=2))
        return
    (args.output_dir / 'config.json').write_text(json.dumps(config, indent=2))
    (args.output_dir / 'categories.json').write_text(json.dumps(categories, indent=2))
    groups = [
        {'params': [p for n, p in model.named_parameters() if p.requires_grad and 'backbone' not in n]},
        {'params': [p for n, p in model.named_parameters() if p.requires_grad and 'backbone' in n],
         'lr': args.lr_backbone}]
    optimizer = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_drop, gamma=0.1)
    start_epoch = 0
    if checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        torch.set_rng_state(checkpoint['rng']['torch'])
        np.random.set_state(checkpoint['rng']['numpy'])
        random.setstate(checkpoint['rng']['python'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(checkpoint['rng']['cuda'])
    train_loader = DataLoader(train_set,
        batch_sampler=BatchSampler(RandomSampler(train_set), args.batch_size, drop_last=True),
        num_workers=args.num_workers, collate_fn=collate_fn)
    if not len(train_loader):
        raise ValueError('Training set is smaller than --batch_size.')
    best_ap = checkpoint.get('best_ap', -1) if checkpoint else -1
    for epoch in range(start_epoch, args.epochs):
        train_stats = train_one_epoch(model, criterion, train_loader, optimizer, device, epoch, args)
        stats = evaluate(model, postprocessors['bbox'], eval_loader, device)
        scheduler.step()
        improved = stats['coco_bbox'][0] > best_ap
        best_ap = max(best_ap, stats['coco_bbox'][0])
        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                 'scheduler': scheduler.state_dict(), 'epoch': epoch, 'config': config,
                 'categories': categories, 'best_ap': best_ap,
                 'rng': {'torch': torch.get_rng_state(), 'numpy': np.random.get_state(),
                         'python': random.getstate(),
                         'cuda': torch.cuda.get_rng_state_all() if device.type == 'cuda' else []}}
        torch.save(state, args.output_dir / 'checkpoint_last.pth')
        if improved:
            torch.save(state, args.output_dir / 'checkpoint_best.pth')
        if (epoch + 1) % args.save_every == 0:
            torch.save(state, args.output_dir / f'checkpoint_{epoch + 1:04d}.pth')
        with (args.output_dir / 'log.jsonl').open('a') as f:
            f.write(json.dumps({'epoch': epoch, 'train': train_stats, 'val': stats}) + '\n')

if __name__ == '__main__':
    main(get_args_parser().parse_args())
