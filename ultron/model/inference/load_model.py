from peft import PeftModel, PeftConfig
from transformers import AutoModel, AutoTokenizer,LlavaNextForConditionalGeneration,LlavaNextProcessor
from rich.console import Console


def load_visual_model_2(checkpoint_path = "/nfs-shared-2/limuyao/JARVIS/checkpoints/mc-llava_v1.6_mistral_7b-lora-embodied_mini_craft_table-10-04-A40-c4-e3-b4-a2/checkpoint-1749"):
    if "lora" in checkpoint_path:
        config = PeftConfig.from_pretrained(checkpoint_path)
        model_name_or_path = config.base_model_name_or_path
    else:
        model_name_or_path = checkpoint_path
    if 'llava-next' in model_name_or_path or 'llava-v1.6' in model_name_or_path:
    # 加载基础模型
        model = LlavaNextForConditionalGeneration.from_pretrained(model_name_or_path)
        processor = LlavaNextProcessor.from_pretrained(model_name_or_path)
        processor.chat_template = "{% for message in messages %}{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'}}{# Render all images first #}{% for content in message['content'] | selectattr('type', 'equalto', 'image') %}{{ '<image>' }}{% endfor %}{# Render all text next #}{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}{{ '\n' + content['text'] + '<|eot_id|>' }}{% endfor %}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
        LLM_backbone = "llama-3"
        VLM_backbone = "llava-next"
    else:
        raise ValueError("暂未加入该模型")
    # 合并新老模型
    if "lora" in checkpoint_path:
        model = PeftModel.from_pretrained(model, model_id=checkpoint_path)
    Console().log(f"[Kernel]{checkpoint_path}checkpoint成功加载")
    return processor,model,LLM_backbone,VLM_backbone

def load_visual_model_1(model_path="/nfs-shared/models/llama3-llava-next-8b-hf",checkpoint_path = "/nfs-shared-2/limuyao/JARVIS/checkpoints/mc-llava_next_llama3_8b-lora-embodied_mini_craft_table-10-05-A40-c4-e3-b4-a2/checkpoint-1749"):
    from transformers import AutoTokenizer, LlavaNextForConditionalGeneration,AutoProcessor
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LlavaNextForConditionalGeneration.from_pretrained(model_path, device_map=device, torch_dtype = torch.bfloat16)
    processor = AutoProcessor.from_pretrained(model_path)
    LLM_backbone = "llama-3"
    VLM_backbone = "llava-next"
    model.load_adapter(checkpoint_path)
    model.enable_adapters()
    return processor,model,LLM_backbone,VLM_backbone

def load_visual_model(model_path="/nfs-shared/models/llama3-llava-next-8b-hf",checkpoint_path = "/nfs-shared-2/limuyao/JARVIS/checkpoints/mc-llava_next_llama3_8b-lora-embodied_mini_craft_table-10-05-A40-c4-e3-b4-a2/checkpoint-1749"):
    from transformers import AutoModelForCausalLM,AutoProcessor
    import torch
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_path,
        device_map=device, 
    )
    model = PeftModel.from_pretrained(model, checkpoint_path)
    model = model.merge_and_unload()
    processor = AutoProcessor.from_pretrained(model_path)
    LLM_backbone = "llama-3"
    VLM_backbone = "llava-next"
    return processor,model,LLM_backbone,VLM_backbone

if __name__ == "__main__":
    processor,model,LLM_backbone,VLM_backbone = load_visual_model()
    
    