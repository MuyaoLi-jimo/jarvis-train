batch=4
gradient_accumulation_steps=4
logging_step=1
epoch=3
learning_rate=1.4e-5
card_type="A40"
card_number=4
cuda_visible_devices=2,3,4,5
training_port=20010
dataset_name="/nfs-shared/pretrain-jarvis-data/trajectory/embodied_mini_craft_10-10-08-llama-3"
version="mc-llava_next_llama3_8b-lora-embodied-mini-craft-10_10-15"  # {model_version}_{dataset_version}_{training_date}
WANDB_NAME="$version-$card_type-c$card_number-e$epoch-b$batch-a$gradient_accumulation_steps"


export WANDB_API_KEY=998c5dff7e8a2d9fb877a2492f1d9ac46aeda17d
export WANDB_PROJECT=Scaling-Law_VLA_Train
export WANDB_NOTES="[24-10-10] 18668 vision language convs (instruction state action)"

deepspeed --include localhost:$cuda_visible_devices --master_port=$training_port ultron/model/train/vsft.py \
    --dataset_name $dataset_name \
    --model_name_or_path "/nfs-shared/models/llama3-llava-next-8b-hf" \
    --report_to "wandb" \
    --dataloader_num_workers 8 \
    --learning_rate $learning_rate \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --per_device_train_batch_size $batch \
    --per_device_eval_batch_size $batch \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --evaluation_strategy "steps" \
    --eval_steps 100 \
    --save_strategy "steps" \
    --save_steps 400 \
    --save_total_limit 20 \
    --output_dir "/nfs-shared-2/limuyao/JARVIS/checkpoints/$WANDB_NAME" \
    --run_name $WANDB_NAME \
    --logging_strategy "steps" \
    --logging_steps $logging_step \
    --num_train_epochs 3 \
    --gradient_checkpointing \
    --torch_dtype bfloat16 \
    --bf16 True \
    --remove_unused_columns False \
    --max_seq_length 2048 \
    --use_peft True \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_target_modules "all-linear" \
    --deepspeed configs/deepspeed_config_s2.json 
