#!/usr/bin/env bash
# Runs the remaining Post-Lab training runs and the evaluation, saving each run's output to logs/.
# Usage (from lab4/):  bash run_all.sh
set -euo pipefail
cd "$(dirname "$0")"
source ../venv/bin/activate
mkdir -p logs

run() {  # run <log name> <train.py args...>
    local log="logs/$1.log"; shift
    echo "=== $(date '+%H:%M:%S') python train.py $* -> $log"
    python train.py "$@" 2>&1 | tee "$log"
}

run train_student_response --kd response -e 100 --augment
run train_student_feature  --kd feature  -e 100 --augment
run train_custom_none      --model custom --kd none     -e 100 --augment
run train_custom_response  --model custom --kd response -e 100 --augment
run train_custom_feature   --model custom --kd feature  -e 100 --augment
run train_custom_both      --model custom --kd both     -e 100 --augment

echo "=== $(date '+%H:%M:%S') python evaluate.py -> logs/evaluate.log"
python evaluate.py 2>&1 | tee logs/evaluate.log
echo "=== All done $(date '+%H:%M:%S')"
