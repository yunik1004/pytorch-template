#!/bin/sh

# Private variables
SCRIPT_DIR=$(dirname $(dirname $(realpath "$0")))/"scripts"

# Environment Setting
set -x  # print the commands

export CUDA_VISIBLE_DEVICES=0

# Arguments
LOG_DIR="."
SEED=0
LR=1e-3
MAX_EPOCHS=10
VAL_EPOCH_FREQ=1
BATCH_SIZE=8
NUM_WORKERS=0
PY_ARGS=${@}  # Additional args

# Run the script
python ${SCRIPT_DIR}/"train.py" \
    --log_dir ${LOG_DIR} \
    --seed ${SEED} \
    --lr ${LR} \
    --max_epochs ${MAX_EPOCHS} \
    --val_epoch_freq ${VAL_EPOCH_FREQ} \
    --batch_size ${BATCH_SIZE} \
    --num_workers ${NUM_WORKERS} \
    ${PY_ARGS}
