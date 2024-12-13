# 全参数微调

batch=16
gradient_accumulation_steps=4
logging_step=1
epoch=20
learning_rate=1.4e-4
card_type="A100"
card_number=4
cuda_visible_devices=0,1,2,3
training_port=20014
dataset_name="/home/limuyao/datas/jarvis-dataset-002/embodied_mini_10-23-craft-craft_table-shell_agent-easy-mistrall"
version="mc-llava_v1.6_mistral_7b-full-embodied_mini_10-23-craft-craft_table-shell_agent-easy-mistral-10-24"  # {model_version}_{dataset_version}_{training_date}"
WANDB_NAME="$version-$card_type-c$card_number-e$epoch-b$batch-a$gradient_accumulation_steps"

export WANDB_API_KEY=998c5dff7e8a2d9fb877a2492f1d9ac46aeda17d
export WANDB_PROJECT=Scaling-Law_VLA_Train
export WANDB_NOTES="[24-10-04] 18668 vision language convs (instruction state action)"

deepspeed --include localhost:$cuda_visible_devices --master_port=$training_port ultron/model/train/vsft.py \
    --dataset_name $dataset_name \
    --model_name_or_path "/nfs-shared/models/llava-v1.6-mistral-7b-hf" \
    --report_to "wandb" \
    --learning_rate $learning_rate \
    --weight_decay 0. \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --per_device_train_batch_size $batch \
    --per_device_eval_batch_size $batch \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --evaluation_strategy "steps" \
    --eval_steps 10 \
    --save_strategy "steps" \
    --save_steps 400 \
    --save_total_limit 20 \
    --output_dir "/nfs-shared/limuyao2/JARVIS/checkpoints/$WANDB_NAME" \
    --run_name $WANDB_NAME \
    --logging_strategy "steps" \
    --logging_steps $logging_step \
    --num_train_epochs epoch \
    --gradient_checkpointing \
    --torch_dtype bfloat16 \
    --bf16 True \
    --remove_unused_columns False \
    --max_seq_length 2048 \
    --deepspeed configs/deepspeed_config_s2.json 