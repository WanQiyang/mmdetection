#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=1

SCRIPT_NAME=0410_01

python -u tools/train.py \
    workspace/configs/retinanet_cub_12.py \
    --work-dir workspace/logs/${SCRIPT_NAME}
