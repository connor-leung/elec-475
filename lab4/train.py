import argparse
import os
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.custom import CompactSegNet
from models.student import StudentNet
from utils.dataset import get_voc
from utils.device import get_device, device_name, synchronize
from utils.kd import response_kd_loss, feature_kd_loss
from utils.metrics import update_confusion, mean_iou
from utils.teacher import load_teacher, teacher_forward

MODELS = {'student': StudentNet, 'custom': CompactSegNet}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=list(MODELS), default='student',
                        help='student = provided StudentNet, custom = our CompactSegNet (Step 3)')
    parser.add_argument('--kd', choices=['none', 'response', 'feature', 'both'], default='none')
    parser.add_argument('--data', default='./data')
    parser.add_argument('-e', '--epochs', type=int, default=3)
    parser.add_argument('-b', '--batch', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--size', type=int, default=320)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--tau', type=float, default=4.0)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--lr-patience', type=int, default=2)
    parser.add_argument('--lr-factor', type=float, default=0.5)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--tol', type=float, default=0.1)
    parser.add_argument('--augment', action='store_true',
                        help='random horizontal flip + random scale/crop on the training images')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--iters', type=int, default=0,
                        help='kick-start: stop after this many iterations (0 = full training)')
    args = parser.parse_args()

    torch.manual_seed(0)
    device = get_device()
    train_loader = DataLoader(get_voc(args.data, 'train', args.size, args.augment),
                              batch_size=args.batch, shuffle=True, num_workers=args.workers,
                              drop_last=True, persistent_workers=args.workers > 0)
    val_loader = DataLoader(get_voc(args.data, 'val', args.size), batch_size=args.batch,
                            shuffle=False, num_workers=args.workers,
                            persistent_workers=args.workers > 0)

    student = MODELS[args.model]().to(device)
    params = list(student.parameters())
    teacher = load_teacher(device) if args.kd != 'none' else None
    # Output files: student_<kd>.pth / loss_<kd>.png for StudentNet, custom_<kd> for CompactSegNet
    tag = f'{args.kd}' if args.model == 'student' else f'{args.model}_{args.kd}'
    ckpt = f'checkpoints/{"student" if args.model == "student" else args.model}_{args.kd}.pth'

    # 1x1 adapters map student features to the teacher's channel counts (training only)
    adapters = None
    if args.kd in ('feature', 'both'):
        ch = student.feat_channels
        adapters = nn.ModuleDict({'low': nn.Conv2d(ch['low'], 512, 1),
                                  'high': nn.Conv2d(ch['high'], 2048, 1)}).to(device)
        params += list(adapters.parameters())

    optimizer = torch.optim.Adam(params, lr=args.lr)
    # Halve the learning rate when the validation mIoU (in points) stops improving
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=args.lr_factor, patience=args.lr_patience,
        threshold=args.tol, threshold_mode='abs')
    print(f'Training {args.model} ({sum(p.numel() for p in student.parameters()):,} parameters) '
          f'with kd={args.kd} on {device_name(device)}' +
          (f' for {args.iters} iterations' if args.iters else f' for up to {args.epochs} epochs'))
    print(vars(args))

    losses, val_mious = [], []
    best_miou, best_epoch, stale = -1.0, 0, 0
    start = time.time()
    for epoch in range(1, (10**6 if args.iters else args.epochs) + 1):
        student.train()
        for img, mask in train_loader:
            img, mask = img.to(device), mask.to(device)
            out = student(img)
            if args.kd == 'none':
                loss = F.cross_entropy(out['out'], mask, ignore_index=255)
            else:
                with torch.no_grad():
                    z_t, f_t = teacher_forward(teacher, img)
                if args.kd == 'feature':
                    loss = F.cross_entropy(out['out'], mask, ignore_index=255)
                else:
                    loss = response_kd_loss(out['out'], z_t, mask, args.alpha, args.beta, args.tau)
                if args.kd in ('feature', 'both'):
                    loss = loss + args.gamma * (
                        feature_kd_loss(adapters['low'](out['low']), f_t['layer2']) +
                        feature_kd_loss(adapters['high'](out['high']), f_t['layer4']))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if args.iters:
                if len(losses) % 5 == 0:
                    print(f'iter {len(losses):3d}/{args.iters}  loss {loss.item():.3f}')
                if len(losses) >= args.iters:
                    break
        if args.iters:
            if len(losses) >= args.iters:
                break
            continue

        student.eval()
        conf = torch.zeros(21, 21, dtype=torch.long)
        with torch.no_grad():
            for img, mask in val_loader:
                pred = student(img.to(device))['out'].argmax(1).cpu()
                conf = update_confusion(conf, pred, mask)
        miou = 100 * mean_iou(conf)
        val_mious.append(miou)
        lr = optimizer.param_groups[0]['lr']
        synchronize(device)
        # Early stopping: an epoch is an improvement only if the mIoU rises by more than --tol
        improved = miou > best_miou + args.tol
        if improved:
            best_miou, best_epoch, stale = miou, epoch, 0
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(student.state_dict(), ckpt)
        else:
            stale += 1
        print(f'epoch {epoch}/{args.epochs}  loss {sum(losses[-len(train_loader):]) / len(train_loader):.3f}'
              f'  val mIoU {miou:.1f}{" *" if improved else ""}  lr {lr:.2e}'
              f'  ({time.time() - start:.0f} s)', flush=True)
        scheduler.step(miou)
        if stale >= args.patience:
            print(f'Early stopping: no improvement of more than {args.tol} mIoU '
                  f'for {args.patience} epochs')
            break

    os.makedirs('plots', exist_ok=True)
    if args.iters:
        first, last = sum(losses[:5]) / 5, sum(losses[-5:]) / 5
        print(f'Kick-start finished: average loss {first:.3f} (first 5) -> {last:.3f} (last 5)')
    else:
        print(f'Finished after {len(val_mious)} epochs in {time.time() - start:.0f} s: '
              f'best val mIoU {best_miou:.1f} at epoch {best_epoch}')
        print(f'Saved {ckpt}')
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(losses, color='tab:blue', linewidth=0.8)
    ax.set_xlabel('iteration')
    ax.set_ylabel('training loss', color='tab:blue')
    if val_mious:
        # Validation mIoU after every epoch, on a second axis
        ax2 = ax.twinx()
        ax2.plot([len(train_loader) * (i + 1) for i in range(len(val_mious))], val_mious,
                 color='tab:orange', marker='o', markersize=3)
        ax2.set_ylabel('validation mIoU (%)', color='tab:orange')
    ax.set_title(f'Training loss ({args.model}, kd={args.kd})')
    fig.tight_layout()
    fig.savefig(f'plots/loss_{tag}.png', dpi=100)
    print(f'Saved plots/loss_{tag}.png')


if __name__ == '__main__':
    main()
