#!/bin/bash

cuda_visible_devices="6,7"
card_num=2
port=9207
workers=30
max_frames=600
temperature=1
env_file="craft_crafting_table_multi"
env_config="jarvis-rt2/$env_file"
base_model_path="/scratch/mc_lmy/models/llama3-llava-next-8b-hf"
model_local_path="llama3-llava-next-8b-hf-craft-craft_table-shell_agent-hard-llama-3-11-17-1-A100-c4-e3-b16-a4"

for checkpoint in 1700 1800 1900 2000; do  #100 200
    echo "Running for checkpoint $checkpoint..."

    lora_path="/scratch/mc_lmy/models/JARVIS/checkpoints/$model_local_path/checkpoint-$checkpoint"
    model_name_or_path="/scratch/mc_lmy/models/$model_local_path-$checkpoint"
    log_path_name="$model_local_path-$checkpoint-$env_file"

    # Apply LoRA
    python ultron/model/construct/apply_lora.py \
        --lora-path $lora_path \
        --base-model-path $base_model_path \
        --enable-processor True \
        --target-model-path $model_name_or_path

    # Serve vLLM
    python ultron/model/inference/serve_vllm.py \
        --card-num $card_num \
        --cuda-visible-devices $cuda_visible_devices \
        --port $port \
        --model-path $model_name_or_path \
        --start True

    # Evaluate
    python mc_evaluate/rollout/agent_evaluate.py \
        --workers $workers \
        --env-config $env_config \
        --max-frames $max_frames \
        --temperature $temperature \
        --checkpoints $model_name_or_path \
        --api_base "http://localhost:$port/v1" 
        #--verbos True

    # Optionally, stop the server (if needed)
    python ultron/model/inference/serve_vllm.py \
        --model-path $model_name_or_path

    # Cleanup
    rm -rf $model_name_or_path

    # Copy results to a temporary folder
    mkdir -p mc_evaluate/temp/$log_path_name
    cp /scratch/mc_lmy/evaluate/$log_path_name/0.mp4 mc_evaluate/temp/$log_path_name/0.mp4
    cp /scratch/mc_lmy/evaluate/$log_path_name/end.json mc_evaluate/temp/$log_path_name/end.json
    cp /scratch/mc_lmy/evaluate/$log_path_name/image.png mc_evaluate/temp/$log_path_name/image.png
    sleep 30
done