# 全参数微调

batch=16
gradient_accumulation_steps=1
logging_step=1
epoch=10
learning_rate=1.4e-5
card_type="A100"
card_number=4
cuda_visible_devices=0,1,2,3
training_port=20012
dataset_name="/home/limuyao/datas/jarvis-dataset-003/embodied_mini_10-30-craft-craft_table-shell_agent-normal-mistral"
version="mc-molmo-7b-d-0924-full-embodied_mini_10-30-craft-craft_table-shell_agent-normal-mistral-11-3"  # {model_version}_{dataset_version}_{training_date}"
WANDB_NAME="$version-$card_type-c$card_number-e$epoch-b$batch-a$gradient_accumulation_steps"

export WANDB_MODE="offline"
export WANDB_API_KEY=998c5dff7e8a2d9fb877a2492f1d9ac46aeda17d
export WANDB_PROJECT=Scaling-Law_VLA_Train
export WANDB_NOTES="[24-11-3] 4.75k vision language convs convs (instruction state action)"

deepspeed --include localhost:$cuda_visible_devices --master_port=$training_port ultron/model/train/vsft.py \
    --dataset_name $dataset_name \
    --dataloader_num_workers 8 \
    --model_name_or_path "/scratch/models/molmo-7b-d-0924" \
    --trust_remote_code True \
    --report_to "wandb" \
    --learning_rate $learning_rate \
    --weight_decay 0. \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --per_device_train_batch_size $batch \
    --per_device_eval_batch_size $batch \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --evaluation_strategy "steps" \
    --eval_steps 72 \
    --save_strategy "steps" \
    --save_steps 72 \
    --save_total_limit 20 \
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
    --deepspeed configs/deepspeed_config_s2.json 