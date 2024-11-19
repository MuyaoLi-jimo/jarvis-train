from transformers import AutoModelForCausalLM,LlavaNextForConditionalGeneration
from ultron.model.train.utils import print_trainable_parameters
model = LlavaNextForConditionalGeneration.from_pretrained(
    "/scratch/mc_lmy/models/mc-llava_next_llama3_8b-LORA-embodied_v4_8_28-8_29-A800-c8-e3-b4-a4",
    trust_remote_code = True,
)

print_trainable_parameters(model,f"model_structure.json")