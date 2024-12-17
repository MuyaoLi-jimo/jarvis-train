#!/bin/bash

cuda_visible_devices="5,6"
card_num=2
port=9213
workers=30
max_frames=600
temperature=1
env_file="craft_birch_planks_multi"
env_config="jarvis-rt2/$env_file"
history_num=0
action_chunk_len=1
bpe=0
instruction_type="recipe" 
base_model_path="/public/models/llama3-llava-next-8b-hf" # llama3-llava-next-8b-hf"  #"/scratch/mc_lmy/models/llama3-llava-next-8b-hf"  #/scratch/mc_lmy/models/JARVIS/checkpoints/mc_llama3-llava-next-8b-hf-full-11-25-craft-10-shell_agent-hard-llama-3-h0-11-25-1-A100-c8-e3-b16-a4/checkpoint-1200
model_local_path="mc_llama3-llava-next-8b-hf-full-11-27-craft-birch_planks-shell_agent-hard-llama-3-h0-c1-12-14-1-A100-c8-e3-b16-a4"

for checkpoint in 100 200 300; do  #800 900 1000 1100 1300
    echo "Running for checkpoint $checkpoint..."

    checkpoint_path="/scratch/limuyao/models/JARVIS/checkpoints/$model_local_path/checkpoint-$checkpoint"
    model_name_or_path="/scratch/limuyao/models/$model_local_path-$checkpoint"
    log_path_name="$model_local_path-$checkpoint-$env_file"

    # Apply LoRA
    if [[ "$model_local_path" == *"lora"* ]]; then
        echo "Model path contains 'lora', proceeding with LoRA application..."
        python ultron/model/construct/apply_lora.py \
            --lora-path $checkpoint_path \
            --base-model-path $base_model_path \
            --enable-processor True \
            --target-model-path $model_name_or_path
    else 
        echo "Model path does not contain 'lora', adding config to the checkpoint"
        model_name_or_path=$checkpoint_path
        python ultron/model/construct/apply_full_sft_vlm.py \
            --base-model-path $base_model_path \
            --enable-processor True \
            --sft-model-path $model_name_or_path
    fi

    # Serve vLLM
    python ultron/model/inference/serve_vllm.py \
        --card-num $card_num \
        --cuda-visible-devices $cuda_visible_devices \
        --max-model-len 3072 \
        --limit-mm-per-prompt 4 \
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
        --api-base "http://localhost:$port/v1" \
        --history-num $history_num \
        --instruction-type $instruction_type \
        --action-chunk-len $action_chunk_len \
        --bpe $bpe\
        #--verbos True \
        
        

    # Optionally, stop the server (if needed)
    python ultron/model/inference/serve_vllm.py \
        --model-path $model_name_or_path

    # Cleanup if lora
    if [[ "$model_local_path" == *"lora"* ]]; then
        echo "clean the model $model_name_or_path"
        rm -rf $model_name_or_path
    else
        echo "do not need to clean the model"
    fi

    # Copy results to a temporary folder
    mkdir -p mc_evaluate/record/success_rate/$log_path_name
    cp /scratch/limuyao/evaluate/$log_path_name/0.mp4 mc_evaluate/record/success_rate/$log_path_name/0.mp4
    cp /scratch/limuyao/evaluate/$log_path_name/end.json mc_evaluate/record/success_rate/$log_path_name/end.json
    cp /scratch/limuyao/evaluate/$log_path_name/image.png mc_evaluate/record/success_rate/$log_path_name/image.png
    sleep 90
done