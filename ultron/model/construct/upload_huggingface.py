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
        folder_path="/scratch/mc_lmy/models/JARVIS/checkpoints/mc_llama3-llava-next-8b-hf-full-11-25-craft-10-shell_agent-hard-llama-3-h0-11-25-1-A100-c8-e3-b16-a4/checkpoint-1200",
        path_in_repo="limuyu011/mc_llama3-llava-next-8b-hf-full-11-25-craft-10-shell_agent-hard-llama-3-h0-11-25-1-A100-1200",
        repo_id="limuyu011/mc_models",
        repo_type="model",
    )
    
    from huggingface_hub import HfApi

def upload_file(file_name="/home/mc_lmy/workspace/mark2.tar.gz"):
    api = HfApi()
    api.upload_file(
        path_or_fileobj="/scratch/mc_lmy/models/JARVIS/checkpoints/mc_llama3-llava-next-8b-hf-full-11-25-craft-10-shell_agent-hard-llama-3-h0-11-25-1-A100-c8-e3-b16-a4/checkpoint-1200",
        path_in_repo="mc_llama3-llava-next-8b-hf-full-11-25-craft-10-shell_agent-hard-llama-3-h0-11-25-1-A100-c8-e3-b16-a4-1200.tar.gz",  # 这里可以指定在仓库中的路径和文件名
        repo_id="limuyu011/mc_models",
        repo_type="model"
    )

if __name__ == "__main__":
    my_list = [
        "/scratch/mc_lmy/models/JARVIS/checkpoints/mc_llama3-llava-next-8b-hf-full-11-25-craft-10-shell_agent-hard-llama-3-h0-11-25-1-A100-c8-e3-b16-a4/checkpoint-1200",
    ]
    #upload_file()
    #exit()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/scratch/mc_lmy/models/JARVIS/checkpoints/mc_llama3-llava-next-8b-hf-full-11-25-craft-10-shell_agent-hard-llama-3-h0-11-25-1-A100-c8-e3-b16-a4/checkpoint-1200")
    parser.add_argument("--enable-processor", type=bool, default=True)
    parser.add_argument("--hub_id", type=str, default="limuyu011/mc_llama3-llava-next-8b-hf-full-11-25-craft-10-shell_agent-hard-llama-3-h0-11-25-1-A100-1200")

    args = parser.parse_args()

    
    upload_model(args.model_path,args.enable_processor,args.hub_id)