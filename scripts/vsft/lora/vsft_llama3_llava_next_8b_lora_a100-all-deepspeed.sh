# MSG：很容易爆显存

batch=16
gradient_accumulation_steps=4
logging_step=1
epoch=2
learning_rate=1.4e-5
card_type="A100"
card_number=4
cuda_visible_devices=4,5,6,7
training_port=20002
dataset_name="/scratch/mc_lmy/datas/11-10-craft-craft_table-shell_agent-hard/output/11-10-craft-craft_table-shell_agent-hard-llama-3"
max_seq_length=2048
version="mc_llama3-llava-next-8b-hf-lora-craft-craft_table-shell_agent-hard-llama-3-11-22-1"  # {model_version}_{dataset_version}_{training_date}"
WANDB_NAME="$version-$card_type-c$card_number-e$epoch-b$batch-a$gradient_accumulation_steps"
# #/scratch/mc_lmy/models/llama3-llava-next-8b-hf" \
# llama3-llava-next-8b-hf-craft-craft_table-shell_agent-hard-llama-3-11-16-1-A100-c4-e3-b16-a1
export WANDB_MODE="offline"
export WANDB_API_KEY=998c5dff7e8a2d9fb877a2492f1d9ac46aeda17d
export WANDB_PROJECT=Scaling-Law_VLA_Train
export WANDB_NOTES="[24-11-22] 219.4k vision language convs convs (instruction state action),with image augment"

deepspeed --include localhost:$cuda_visible_devices --master_port=$training_port ultron/model/train/vsft.py \
    --dataset_name $dataset_name \
    --model_name_or_path "/scratch/mc_lmy/models/llama3-llava-next-8b-hf" \
    --dataloader_num_workers 16 \
    --remove_unused_columns False \
    --report_to "wandb" \
    --learning_rate $learning_rate \
    --weight_decay 0. \
    --max_grad_norm 10 \
    --warmup_ratio 0.16 \
    --warmup_steps 400 \
    --lr_scheduler_type "cosine" \
    --per_device_train_batch_size $batch \
    --per_device_eval_batch_size $batch \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --evaluation_strategy "steps" \
    --eval_steps 100 \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 30 \
    --output_dir "/scratch/mc_lmy/models/JARVIS/checkpoints/$WANDB_NAME" \
    --run_name $WANDB_NAME \
    --logging_strategy "steps" \
    --logging_steps $logging_step \
    --num_train_epochs $epoch \
    --gradient_checkpointing \
    --torch_dtype bfloat16 \
    --bf16 True \
    --remove_unused_columns False \
    --max_seq_length $max_seq_length \
    --use_peft True \
    --lora_r 64 \
    --lora_alpha 16 \
    --private_lora_structure \
    --deepspeed configs/deepspeed_config_s2.json 
    #--lora_modules_to_save 'embed_tokens' \
    
