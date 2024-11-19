from peft import PeftModel, PeftConfig
from transformers import AutoModel, AutoTokenizer
from rich.console import Console
import torch


def load_visual_model(device,checkpoint_path = "/nfs-shared-2/limuyao/JARVIS/checkpoints/mc-llava_v1.6_mistral_7b-LORA-embodied_mini_craft_table-10-04-A40-c4-e3-b4-a2/checkpoint-1749",quick_load=False):
    
    if "mistral" in checkpoint_path:
        LLM_backbone = "mistral"
    elif "vicuna" in checkpoint_path:
        LLM_backbone = "llama-2"
    else:
        LLM_backbone = "llama-3"
    if 'llava-next' in checkpoint_path or 'llava_next' in checkpoint_path or 'llava-v1.6' in checkpoint_path or 'llava_v1.6'  in checkpoint_path:
        VLM_backbone = "llava-next"
    
    if quick_load:
        return LLM_backbone,VLM_backbone
    if "lora" in checkpoint_path:
        config = PeftConfig.from_pretrained(checkpoint_path)
        model_name_or_path = config.base_model_name_or_path
    else:
        model_name_or_path = checkpoint_path
    if VLM_backbone == "llava-next":
        from transformers import LlavaNextForConditionalGeneration,LlavaNextProcessor
    # 加载基础模型
        model = LlavaNextForConditionalGeneration.from_pretrained(model_name_or_path).to(device)
        processor = LlavaNextProcessor.from_pretrained(model_name_or_path)
        processor.chat_template = "{% for message in messages %}{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'}}{# Render all images first #}{% for content in message['content'] | selectattr('type', 'equalto', 'image') %}{{ '<image>' }}{% endfor %}{# Render all text next #}{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}{{ '\n' + content['text'] + '<|eot_id|>' }}{% endfor %}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
    else:
        raise ValueError("暂未加入该模型")
    # 合并新老模型
    if "lora" in checkpoint_path:
        model = PeftModel.from_pretrained(model, model_id=checkpoint_path).to(device)
        model = model.merge_and_unload()
    Console().log(f"[Kernel]{checkpoint_path} 模型成功加载")
    return processor,model,LLM_backbone,VLM_backbone

if __name__ == "__main__":
    processor,model,LLM_backbone,VLM_backbone = load_visual_model()
    
    