#! /bin/bash

cuda_visible_devices=0,1,2,3
card_num=4
port=9206 #9205  #9202
model_name_or_path="/public/zhwang/checkpoints/mc-sft-llava_v1.6_vicuna_7b-mcqa_v3_12_25_277k-11_19-A100-c8-e1-b8-a1"
#/scratch/mc_lmy/models/mc-llava_next_llama3_8b-LORA-embodied_v4_8_28-8_29-A800-c8-e3-b4-a4-craft-craft_table-shell_agent-hard-llama-3-11-13-2-A100-c4-e3-b16-a4-1281
#/scratch/mc_lmy/models/mc-llava_next_llama3_8b-LORA-11-10-craft-craft_table-shell_agent-hard-llama-3-11-13-2-A100-c4-e3-b16-a4-1281

CUDA_VISIBLE_DEVICES=$cuda_visible_devices vllm serve $model_name_or_path \
    --port $port \
    --max-model-len 4096 \
    --max-num-seqs 32 \
    --gpu-memory-utilization 0.95 \
    --tensor-parallel-size $card_num \
    --trust-remote-code \
    --limit-mm-per-prompt image=4 \
    #--chat-template /scratch2/limuyao/workspace/VLA_benchmark/data/model/template/template_llava.jinja \