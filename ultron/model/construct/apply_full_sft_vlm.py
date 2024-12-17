import argparse
from pathlib import Path
import shutil


def apply_full_model(args):
    base_model_path = args.base_model_path
    if args.enable_processor:
        if 'llava-next' in base_model_path or 'llava-v1.6' in base_model_path:
            from transformers import LlavaNextProcessor
            processor = LlavaNextProcessor.from_pretrained(base_model_path)
            processor.save_pretrained(args.sft_model_path)
            if "vicuna" in base_model_path.name:
                config_file = base_model_path / "preprocessor_config.json"
                target_path = args.sft_model_path / "preprocessor_config.json"
                
                if config_file.exists():
                    shutil.copy(str(config_file), str(target_path))
                    print(f"Copied preprocessor_config.json from {config_file} to {target_path}")
                else:
                    print(f"Warning: {config_file} does not exist and could not be copied.")
        
                
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=str, default="/public/models/llava-v1.6-vicuna-13b-hf")
    parser.add_argument("--enable-processor", type=bool, default= True,)
    parser.add_argument("--sft-model-path", type=str, default="mc-sft-llava_v1.6_vicuna_13b-mcqa_v3_12_25_277k-12_17-A100-c8-e1-b8-a1/checkpoint-4312")
    args = parser.parse_args()
    apply_full_model(args)