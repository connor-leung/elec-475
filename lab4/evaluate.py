import argparse
import json
import os
import torch
from torch.utils.data import DataLoader

from models.custom import CompactSegNet
from models.student import StudentNet
from utils.dataset import get_voc
from utils.device import get_device, device_name
from utils.metrics import update_confusion, iou_per_class, mean_iou
from utils.profiler import ms_per_image
from utils.teacher import load_teacher
from utils.visualization import save_comparison

CHECKPOINTS = [  # (checkpoint, name in the table, model class)
    ('checkpoints/student_none.pth', 'Student, no KD', StudentNet),
    ('checkpoints/student_response.pth', 'Student, response KD', StudentNet),
    ('checkpoints/student_feature.pth', 'Student, feature KD', StudentNet),
    ('checkpoints/custom_none.pth', 'Custom, no KD', CompactSegNet),
    ('checkpoints/custom_response.pth', 'Custom, response KD', CompactSegNet),
    ('checkpoints/custom_feature.pth', 'Custom, feature KD', CompactSegNet),
    ('checkpoints/custom_both.pth', 'Custom, resp.+feat. KD', CompactSegNet),
]


def image_miou(pred, mask):
    """mIoU of a single image, over the classes that appear in its prediction or ground truth."""
    conf = update_confusion(torch.zeros(21, 21, dtype=torch.long), pred, mask)
    return mean_iou(conf)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='./data')
    parser.add_argument('--size', type=int, default=320)
    parser.add_argument('-b', '--batch', type=int, default=16)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    device = get_device()
    cpu = torch.device('cpu')
    print(f'Evaluating on {device_name(device)}')
    dataset = get_voc(args.data, 'val', args.size)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=args.workers)

    models = {'Teacher (FCN-ResNet50)': load_teacher(device)}
    for path, name, cls in CHECKPOINTS:
        if os.path.exists(path):
            m = cls(pretrained=False).to(device).eval()
            m.load_state_dict(torch.load(path, map_location=device))
            models[name] = m
        else:
            print(f'Skipping {name}: {path} not found')

    rows, preds, masks = [], {}, []
    with torch.no_grad():
        for name, model in models.items():
            conf = torch.zeros(21, 21, dtype=torch.long)
            preds[name] = []
            for img, mask in loader:
                pred = model(img.to(device))['out'].argmax(1).cpu()
                conf = update_confusion(conf, pred, mask)
                preds[name].append(pred.to(torch.uint8))
                if not rows:  # ground truth: collected once, with the first model
                    masks.append(mask)
            preds[name] = torch.cat(preds[name])
            params = sum(p.numel() for p in model.parameters())
            rows.append({'model': name, 'miou': 100 * mean_iou(conf), 'params': params,
                         'ms_per_img': ms_per_image(model, device, args.size),
                         'ms_per_img_cpu': ms_per_image(model.to(cpu), cpu, args.size, runs=20),
                         'class_iou': [100 * v for v in iou_per_class(conf).tolist()]})
            model.to(device)
    masks = torch.cat(masks)

    print(f'\n{"Model":<26}{"mIoU":>8}{"Params":>14}{"ms/img":>10}{"ms/img (CPU)":>14}')
    for r in rows:
        print(f'{r["model"]:<26}{r["miou"]:>8.1f}{r["params"]:>14,}{r["ms_per_img"]:>10.1f}'
              f'{r["ms_per_img_cpu"]:>14.1f}')
    os.makedirs('results', exist_ok=True)
    with open('results/comparison.json', 'w') as f:
        json.dump({'device': device_name(device), 'size': args.size, 'rows': rows}, f, indent=2)
    print('\nSaved results/comparison.json')

    def save(indices, path):
        images = torch.stack([dataset[i][0] for i in indices])
        save_comparison(images, masks[indices], {n: p[indices].long() for n, p in preds.items()}, path)
        print(f'Saved {path} (validation images {indices})')

    save([0, 1, 2, 3], 'plots/compare.png')
    # Extra examples: the best and worst images of the last (best custom) model, by per-image mIoU.
    # Images that are almost all background are left out so the examples contain real objects.
    last = preds[rows[-1]['model']].long()
    scores = torch.tensor([image_miou(last[i], masks[i]) if
                           ((masks[i] > 0) & (masks[i] < 255)).float().mean() > 0.1 else float('nan')
                           for i in range(len(masks))])
    order = torch.argsort(torch.nan_to_num(scores, nan=-1.0), descending=True)
    order = order[:int((~scores.isnan()).sum())]
    save(order[:4].tolist(), 'plots/compare_success.png')
    save(order[-4:].tolist(), 'plots/compare_failure.png')


if __name__ == '__main__':
    main()
