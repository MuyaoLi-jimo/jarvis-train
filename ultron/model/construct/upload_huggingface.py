from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import argparse
from huggingface_hub import HfApi

def upload_model(model_path,enable_processor,hub_id):
    """
    model_path: the model after lora/pretrained
    """
    if 'llava-next' in model_path or 'llava-v1.6' in model_path:
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
        )
        if enable_processor:
            processor = LlavaNextProcessor.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    
    if hub_id:
        model.push_to_hub(hub_id)
        tokenizer.push_to_hub(hub_id)
        if enable_processor:
            processor.push_to_hub(hub_id)

def upload_file():
    
    api = HfApi()
    api.upload_folder(
        folder_path="/home/mc_lmy/model/mc-llava_v1.6_vicuna_7b-lora-embodied_mini_craft_10-10-08-llava-v1.6-A100-c4-e3-b16-a4",
        repo_id="limuyu011/jarvis-001",
        repo_type="model",
    )

if __name__ == "__main__":
    upload_file()
    exit()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/home/mc_lmy/model/mc-llava_v1.6_vicuna_7b-lora-embodied_mini_craft_10-10-08-llava-v1.6-A100-c4-e3-b16-a4")
    parser.add_argument("--enable-processor", type=bool, default=True)
    parser.add_argument("--hub_id", type=str, default="limuyu011/jarvis-001")

    args = parser.parse_args()

    
    upload_model(args.model_path,args.enable_processor,args.hub_id)