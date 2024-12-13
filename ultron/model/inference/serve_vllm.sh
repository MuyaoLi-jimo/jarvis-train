#! /bin/bash

cuda_visible_devices=4,7
card_num=2
port=9206 #9205  #9202
model_name_or_path="/scratch/mc_lmy/models/JARVIS/checkpoints/mc_llama3-llava-next-8b-hf-full-11-27-craft-carrot_on_a_stick-shell_agent-hard-llama-3-h0-12-1-1-A100-c4-e3-b16-a4/checkpoint-1000"
#"/scratch/mc_lmy/models/JARVIS/checkpoints/mc_llava_v1.6_vicuna_7b-full-11-10-craft-craft_table-shell_agent-hard-llama-2-h0-c1-b512-12-08-1-A100-c4-e3-b16-a4/checkpoint-2000"
#/scratch/mc_lmy/models/mc-llava_next_llama3_8b-LORA-embodied_v4_8_28-8_29-A800-c8-e3-b4-a4-craft-craft_table-shell_agent-hard-llama-3-11-13-2-A100-c4-e3-b16-a4-1281
#/scratch/mc_lmy/models/mc-llava_next_llama3_8b-LORA-11-10-craft-craft_table-shell_agent-hard-llama-3-11-13-2-A100-c4-e3-b16-a4-1281

CUDA_VISIBLE_DEVICES=$cuda_visible_devices vllm serve $model_name_or_path \
    --port $port \
    --max-model-len 3000 \
    --max-num-seqs 16 \
    --gpu-memory-utilization 0.95 \
    --tensor-parallel-size $card_num \
    --trust-remote-code \
    --limit-mm-per-prompt image=4 \
    #--chat-template /home/mc_lmy/workspace/jarvis-train/ultron/model/inference/template/template_llava.jinja \
    #--dtype "float32" \
    #--kv-cache-dtype "fp8" \
    #--cpu-offload-gb 10 \
    #--limit-mm-per-prompt image=4 \
       
    