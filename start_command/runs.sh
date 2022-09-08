#!/bin/bash

export PORT=13425
export CUDA_VISIBLE_DEVICES=0,1

cd ..
source activate swin

python -m torch.distributed.launch --nproc_per_node 1 --master_port $PORT main.py \
--data-path "/home/wangxy/datasets/imagewoof" \
--dataset-name "imagewoof" \
--num-classes 10 \
