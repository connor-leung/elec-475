#!/usr/bin/env bash
# ELEC 475 Lab 2 Post-Lab: run every experiment needed for Tables 7.3a / 7.3b.
#
# Usage (from lab2/, with the venv active):
#   ./run_postlab.sh            # run everything
#   ./run_postlab.sh cifar      # only the CIFAR-10 runs
#   ./run_postlab.sh tiny       # only the Tiny ImageNet runs
#   ./run_postlab.sh profile    # only the architecture checks + compare_models
#   ./run_postlab.sh seeds      # extra seeds for the normalization study (Mean ± Std)
#   ./run_postlab.sh extended   # extended ablation (lr, batch size, early stopping)
#
# Afterwards, python summarize_postlab.py prints the Mean ± Std and Δ tables.
#
# A run is skipped if its checkpoint already exists. Delete the .pth to rerun it.
# All output is tee'd to logs/postlab/<name>.log.

set -euo pipefail
cd "$(dirname "$0")"

STAGE="${1:-all}"
SEED=42
EPOCHS=10
AUG_EPOCHS=15
LOG_DIR=logs/postlab
mkdir -p "$LOG_DIR" checkpoints plots

# run <name> <train.py args...>
run() {
    local name="$1"; shift
    local ckpt="checkpoints/${name}.pth"
    if [[ -f "$ckpt" ]]; then
        echo ">>> Skipping ${name} (${ckpt} exists)"
        return
    fi
    echo ">>> Training ${name}"
    # --seed goes first so a run can override it (argparse keeps the last value)
    python -u train.py --seed "$SEED" "$@" \
        --save-model "$ckpt" --plot-file "plots/loss_${name}.png" \
        2>&1 | tee "${LOG_DIR}/${name}.log"
}

# Every model on one dataset. train.py augments by default, so every run except
# the "+ Aug" row passes --no-augment to match the table's "None" column.
run_dataset() {
    local ds="$1" prefix="$2"
    run "${prefix}lenet_baseline"  --model lenet        --dataset "$ds" --epochs "$EPOCHS" --no-augment
    run "${prefix}lenet_modern"    --model lenet_modern --dataset "$ds" --epochs "$EPOCHS" --no-augment
    run "${prefix}alexnet_none"    --model alexnet --norm none --dataset "$ds" --epochs "$EPOCHS" --no-augment
    run "${prefix}alexnet_lrn"     --model alexnet --norm lrn  --dataset "$ds" --epochs "$EPOCHS" --no-augment
    run "${prefix}alexnet_bn"      --model alexnet --norm bn   --dataset "$ds" --epochs "$EPOCHS" --no-augment
    run "${prefix}alexnet_bn_aug"  --model alexnet --norm bn   --dataset "$ds" --epochs "$EPOCHS"
    run "${prefix}zoo_alexnet"     --model zoo_alexnet  --dataset "$ds" --epochs "$EPOCHS" --no-augment
}

if [[ "$STAGE" == all || "$STAGE" == profile ]]; then
    # Step 1: architecture / parameter check (~6.3M params for AlexNet-64)
    python -m models.alexnet 2>&1 | tee "${LOG_DIR}/alexnet_summary.log"
    # Step 5.1: Params / FLOPs / latency table for the report
    python compare_models.py --dataset cifar10       2>&1 | tee "${LOG_DIR}/compare_models_cifar10.log"
    python compare_models.py --dataset tiny_imagenet 2>&1 | tee "${LOG_DIR}/compare_models_tiny_imagenet.log"
fi

if [[ "$STAGE" == all || "$STAGE" == cifar ]]; then
    # Steps 2 + table rows 1-7 on CIFAR-10
    run_dataset cifar10 ""
    # Step 3: generalization gap, with vs without augmentation at 15 epochs
    run alexnet_no_aug   --model alexnet --norm bn --dataset cifar10 --epochs "$AUG_EPOCHS" --no-augment
    run alexnet_with_aug --model alexnet --norm bn --dataset cifar10 --epochs "$AUG_EPOCHS"
fi

if [[ "$STAGE" == all || "$STAGE" == tiny ]]; then
    # Step 4 + Tiny ImageNet columns of Table 7.3b (downloads ~240MB on first run)
    run_dataset tiny_imagenet "tiny_"
fi

if [[ "$STAGE" == all || "$STAGE" == seeds ]]; then
    # Multi-seed normalization study (Mean ± Std). Seed 42 is the alexnet_{none,lrn,bn} run above.
    for s in 0 1; do
        for n in none lrn bn; do
            run "alexnet_${n}_s${s}" --model alexnet --norm "$n" --dataset cifar10 --epochs "$EPOCHS" --no-augment --seed "$s"
        done
    done
fi

if [[ "$STAGE" == all || "$STAGE" == extended ]]; then
    # Extended ablation: each varies one setting from alexnet_bn (lr 1e-3, batch 128, 10 epochs, no aug)
    run alexnet_bn_lr1e-2 --model alexnet --norm bn --dataset cifar10 --epochs "$EPOCHS" --no-augment --lr 1e-2
    run alexnet_bn_lr1e-4 --model alexnet --norm bn --dataset cifar10 --epochs "$EPOCHS" --no-augment --lr 1e-4
    run alexnet_bn_bs64   --model alexnet --norm bn --dataset cifar10 --epochs "$EPOCHS" --no-augment --batch-size 64
    run alexnet_bn_bs256  --model alexnet --norm bn --dataset cifar10 --epochs "$EPOCHS" --no-augment --batch-size 256
    # Longer schedule with augmentation, stopped when val loss stalls for 5 epochs
    run alexnet_bn_aug_es --model alexnet --norm bn --dataset cifar10 --epochs 30 --early-stopping
fi

echo ">>> Done. Logs in ${LOG_DIR}/, checkpoints in checkpoints/, curves in plots/"
