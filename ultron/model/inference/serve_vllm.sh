#! /bin/bash

cuda_visible_devices=4
port=9003
model_name_or_path=/scratch/mc_lmy/models/mc-llava_v1.6_mistral_7b-LORA-embodied_mini_10-30-craft-craft_table-shell_agent-normal-mistral-10-30-A100-c4-e10-b16-a1-576


CUDA_VISIBLE_DEVICES=$cuda_visible_devices vllm serve $model_name_or_path \
    --port $port \
    --max-model-len 4096 \
    --max-num-seqs 16 \
    --gpu-memory-utilization 0.7 \
    --trust-remote-code \
    #--chat-template /scratch2/limuyao/workspace/VLA_benchmark/data/model/template/template_llava.jinja \