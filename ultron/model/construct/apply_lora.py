"""
Apply the LoRA weights on top of a base model.

Usage:
python3 -m fastchat.model.apply_lora --base ~/model_weights/llama-7b --target ~/model_weights/baize-7b --lora project-baize/baize-lora-7B

Dependency:
pip3 install git+https://github.com/huggingface/peft.git@2822398fbe896f25d4dac5e468624dc5fd65a51b
"""
import argparse

import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor


def apply_lora(base_model_path, enable_processor, target_model_path, lora_path):
    print(f"Loading the base model from {base_model_path}")
    # base = AutoModelForCausalLM.from_pretrained(
    #     base_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
    # )
    if 'llava-next' in base_model_path or 'llava-v1.6' in base_model_path or 'llava_next' in base_model_path:
        base_model = LlavaNextForConditionalGeneration.from_pretrained(
            base_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
        )
        if enable_processor:
            processor = LlavaNextProcessor.from_pretrained(base_model_path)
    else:
        # raise ValueError("Unknown model")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
        )
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)

    print(f"Loading the LoRA adapter from {lora_path}")

    lora_model = PeftModel.from_pretrained(
        base_model,
        lora_path,
        # torch_dtype=torch.float16
    )
    print("Applying the LoRA")
    model = lora_model.merge_and_unload()

    print(f"Saving the target model to {target_model_path}")
    model.save_pretrained(target_model_path)
    tokenizer.save_pretrained(target_model_path)
    if enable_processor:
        processor.save_pretrained(target_model_path)

        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=str, default="/scratch/mc_lmy/models/JARVIS/checkpoints/mc_llama3-llava-next-8b-hf-full-craft-craft_table-shell_agent-hard-llama-3-11-22-1-A100-c4-e3-b16-a4/checkpoint-1100")
    parser.add_argument("--lora-path", type=str, default="/scratch/mc_lmy/models/JARVIS/checkpoints/mc_llama3-llava-next-8b-hf-lora-craft-craft_table-shell_agent-hard-llama-3-h1-11-24-1-A100-c8-e1-b8-a4/checkpoint-200")
    parser.add_argument("--enable-processor", type=bool, default= True,)
    #注意不要把lora加进去
    parser.add_argument("--target-model-path", type=str, default="/scratch/mc_lmy/models/mc_llama3-llava-next-8b-hf-LORA-craft-craft_table-shell_agent-hard-llama-3-h1-11-24-1-A100-c8-e1-b8-a4-200")
    
    args = parser.parse_args()

    apply_lora(args.base_model_path, args.enable_processor, args.target_model_path, args.lora_path)