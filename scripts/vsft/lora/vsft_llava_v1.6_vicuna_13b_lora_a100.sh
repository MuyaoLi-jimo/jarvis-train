G：很容易爆显存
# step=data / card number / batch size / gradient_accumulation_steps
batch=16
gradient_accumulation_steps=4
logging_step=1 #2
epoch=3
learning_rate=1.4e-5
card_type="A100"
card_number=4
cuda_visible_devices=0,1,2,3
training_port=20011
dataset_name="/home/limuyao/datas/10-08_craft-10_dataset/embodied_mini_craft_10-10-08-llava-v1.6"
version="mc-llava_v1.6_vicuna_13b-lora-embodied_mini_craft_10-10-08-llava-v1.6"  # {model_version}_{dataset_version}_{training_date}
WANDB_NAME="$version-$card_type-c$card_number-e$epoch-b$batch-a$gradient_accumulation_steps"

export WANDB_API_KEY=998c5dff7e8a2d9fb877a2492f1d9ac46aeda17d
export WANDB_PROJECT=Scaling-Law_VLA_Train
export WANDB_NOTES="[24-10-17] 1329*64*16/3 vision language convs (instruction state action)"

deepspeed --include localhost:$cuda_visible_devices --master_port=$training_port ultron/model/train/vsft.py \
	--dataset_name $dataset_name \
	--model_name_or_path "/home/limuyao/model/llava-v1.6-vicuna-13b-hf" \
	--report_to "wandb" \
	--learning_rate $learning_rate \
	--weight_decay 0. \
	--warmup_ratio 0.03 \
	--lr_scheduler_type "cosine" \
	--per_device_train_batch_size $batch \
	--per_device_eval_batch_size $batch \
	--dataloader_num_workers 8 \
	 --gradient_accumulation_steps $gradient_accumulation_steps \
	 --evaluation_strategy "steps" \
	 --save_strategy "steps" \
	 --eval_steps 100 \
	 --save_steps 400 \
	 --save_total_limit 20 \
	 --output_dir "/home/limuyao/model/JARVIS/checkpoints/$WANDB_NAME" \
	 --run_name $WANDB_NAME \
	 --logging_strategy "steps" \
	 --logging_steps $logging_step \
	 --num_train_epochs 3 \
	--torch_dtype bfloat16 \
	 --gradient_checkpointing \
	 --bf16 True \
	 --remove_unused_columns False \
	--max_seq_length 2048 \
	--use_peft True \
	--lora_r 64 \
	--lora_alpha 16 \
	--lora_target_modules "all-linear" \
	--deepspeed configs/deepspeed_config_s2.json 
