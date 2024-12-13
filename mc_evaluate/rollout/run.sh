cuda_visible_devices=4,5
card_num=2
port=9206
workers=30
max_frames=600
temperature=1
env_config="jarvis-rt2/craft_crafting_table_multi"
base_model_path="/scratch/mc_lmy/models/llama3-llava-next-8b-hf"
lora_path="/scratch/mc_lmy/models/JARVIS/checkpoints/mc-llava_next_llama3_8b-lora-11-10-craft-craft_table-shell_agent-hard-llama-3-11-16-1-A100-c4-e3-b16-a1/checkpoint-6400"
model_name_or_path="/scratch/mc_lmy/models/mc-llava_next_llama3_8b-lora-11-10-craft-craft_table-shell_agent-hard-llama-3-11-16-1-A100-c4-e3-b16-a1-3600"

mc_lmy@js1.blockelite.cn -p 31400  mc_lmy@js1.blockelite.cn

python ultron/model/construct/apply_lora.py \
    --lora-path $lora_path \
    --base-model-path $base_model_path \
    --enable-processor True \
    --target-model-path $model_name_or_path

python ultron/model/inference/serve_vllm.py \
    --card-num $card_num \
    --cuda-visible-devices $cuda_visible_devices \
    --port $port \
    --model-path $model_name_or_path \
    --start True

python mc_evaluate/rollout/agent_evaluate.py \
    --workers $workers \
    --env-config $env_config \
    --max-frames $max_frames \
    --temperature $temperature \
    --checkpoints $model_name_or_path \
    --api_base "http://localhost:$port/v1" \

python ultron/model/inference/serve_vllm.py \
    --start False