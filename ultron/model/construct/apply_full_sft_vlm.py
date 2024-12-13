import argparse
from pathlib import Path


def apply_full_model(args):
    base_model_path = args.base_model_path
    if args.enable_processor:
        if 'llava-next' in base_model_path or 'llava-v1.6' in base_model_path:
            from transformers import LlavaNextProcessor
            processor = LlavaNextProcessor.from_pretrained(base_model_path)
            processor.save_pretrained(args.sft_model_path)

    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=str, default="/home/limuyao/model/llava-v1.6-mistral-7b-hf")
    parser.add_argument("--enable-processor", type=bool, default= True,)
    parser.add_argument("--sft-model-path", type=str, default="/scratch/mc_lmy/models/JARVIS/checkpoints/mc-llava_v1.6_mistral_7b-full-embodied_mini_10-23-craft-craft_table-shell_agent-easy-mistral-10-24-A100-c4-e30-b16-a4/checkpoint-120")
    args = parser.parse_args()
    apply_full_model(args)