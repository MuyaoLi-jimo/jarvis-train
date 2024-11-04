#! /bin/bash

cuda_visible_devices=0,1,2,3
card_num=4
port=9003
model_name_or_path=/scratch/models/molmo-72b-0924


CUDA_VISIBLE_DEVICES=$cuda_visible_devices vllm serve $model_name_or_path \
    --port $port \
    --max-model-len 4096 \
    --max-num-seqs 16 \
    --gpu-memory-utilization 0.7 \
    --tensor-parallel-size $card_num \
    --trust-remote-code \
    #--chat-template /scratch2/limuyao/workspace/VLA_benchmark/data/model/template/template_llava.jinja \