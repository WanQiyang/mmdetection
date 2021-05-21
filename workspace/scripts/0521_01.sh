#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=1,2

SCRIPT_NAME=0521_01
GPUS=2
PORT=29500

python -m torch.distributed.launch --nproc_per_node=${GPUS} --master_port=${PORT} tools/train.py \
    workspace/configs/centernet_cub_12.py \
    --work-dir workspace/logs/${SCRIPT_NAME} \
    --gpus 2 \
    --launcher pytorch
