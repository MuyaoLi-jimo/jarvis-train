# MSG：很容易爆显存

batch=16
gradient_accumulation_steps=4
logging_step=1
epoch=3
learning_rate=1.4e-4
card_type="A100"
card_number=4
cuda_visible_devices=4,5,6,7
training_port=20001
dataset_name="/home/mc_lmy/datas/11-10-craft-craft_table-shell_agent-hard/output/11-10-craft-craft_table-shell_agent-hard-llama-3-m1"
version="mc-llava_next_llama3_8b-LORA-embodied_v4_8_28-8_29-A800-c8-e3-b4-a4-craft-craft_table-shell_agent-hard-llama-3-11-12-2"  # {model_version}_{dataset_version}_{training_date}"
WANDB_NAME="$version-$card_type-c$card_number-e$epoch-b$batch-a$gradient_accumulation_steps"

export WANDB_MODE="offline"
export WANDB_API_KEY=998c5dff7e8a2d9fb877a2492f1d9ac46aeda17d
export WANDB_PROJECT=Scaling-Law_VLA_Train
export WANDB_NOTES="[24-11-20] 22.3k vision language convs convs (instruction state action),with image augment"

deepspeed --include localhost:$cuda_visible_devices --master_port=$training_port ultron/model/train/vsft.py \
    --dataset_name $dataset_name \
    --model_name_or_path "/scratch/mc_lmy/models/mc-llava_next_llama3_8b-LORA-embodied_v4_8_28-8_29-A800-c8-e3-b4-a4" \
    --report_to "wandb" \
    --learning_rate $learning_rate \
    --weight_decay 0. \
    --warmup_ratio 0.16 \
    --lr_scheduler_type "cosine" \
    --per_device_train_batch_size $batch \
    --per_device_eval_batch_size $batch \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --evaluation_strategy "steps" \
    --eval_steps 100 \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 20 \
    --dataloader_num_workers 8 \
    --output_dir "/scratch/mc_lmy/models/JARVIS/checkpoints/$WANDB_NAME" \
    --run_name $WANDB_NAME \
    --logging_strategy "steps" \
    --logging_steps $logging_step \
    --num_train_epochs $epoch \
    --gradient_checkpointing \
    --torch_dtype bfloat16 \
    --bf16 True \
    --remove_unused_columns False \
    --max_seq_length 2048 \
    --use_peft True \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_target_modules 'all-linear' \
    --lora_modules_to_save 'embed_tokens' \
    --deepspeed configs/deepspeed_config_s2.json 

