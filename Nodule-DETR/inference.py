"""Run a release checkpoint on an image; export original-image pixel boxes."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser('Nodule-DETR image inference')
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--image', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path('outputs/prediction.json'))
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    args = p.parse_args()
    import torch
    from PIL import Image
    from main import get_args_parser, load_checkpoint
    from models import build_nodule_detr
    from datasets.coco import make_coco_transforms
    checkpoint = load_checkpoint(args.checkpoint)
    config = get_args_parser().parse_args([])
    vars(config).update(checkpoint['config'])
    config.device = args.device
    model, _, processors = build_nodule_detr(config)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.to(args.device).eval()
    image = Image.open(args.image).convert('RGB')
    tensor, _ = make_coco_transforms('val')(image, None)
    with torch.no_grad():
        outputs, _ = model([tensor.to(args.device)], dn_args=0)
        result = processors['bbox'](outputs, torch.tensor([[image.height, image.width]], device=args.device))[0]
    categories = checkpoint['categories']
    detections = []
    for score, label, box in zip(result['scores'], result['labels'], result['boxes']):
        if score.item() < args.threshold:
            continue
        category = categories[label.item()]
        detections.append({'category_id': category['id'], 'category': category['name'],
                           'score': score.item(), 'box_xyxy': box.tolist()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'image': args.image.name, 'detections': detections}, indent=2))
    print(f'Saved {len(detections)} detections to {args.output}')


if __name__ == '__main__':
    main()
