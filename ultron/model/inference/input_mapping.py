from pathlib import Path
import pathlib
from typing import Union
import base64
import numpy as np
import ultron.model.inference.load_model as load_model
import ultron.model.inference.action_mapping as action_mapping
from PIL import Image

class ProcessorWrapper:
    def __init__(self,processor,model_name= "llava-next"):
        self.processor = processor
        self.model_name = model_name
        self.default_image_size = (672,336)
        
    def create_message(self,role="user",type="image",prompt:str=""):
        if self.model_name=="llava-next":
            if type=="image":
                message = {
                    "role":role,
                    "content":[
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ]
                }
            if type == "text":
                message = {
                    "role":role,
                    "content":[
                        {"type": "text", "text": prompt},
                    ]
                }
        return message
    
    def create_text_input(self,conversations:list):
        text_prompt = self.processor.apply_chat_template(conversations, add_generation_prompt=True)
        return text_prompt
    
    def create_image_input(self,image_pixels=None,image_path:str=""):
        image = image_pixels
        if image_path:
            image = Image.open(image_path)
        if self.model_name=="llava-next":
            image = image.resize(self.default_image_size)
        return image

    

if __name__ == "__main__":
    device = "cuda:0"
    processor,model,LLM_backbone,VLM_backbone = load_model.load_visual_model_2(checkpoint_path=r"/nfs-shared-2/limuyao/JARVIS/checkpoints/llava-next-1")
    processor_wrapper = ProcessorWrapper(processor,model_name="llava-next")
    action_map = action_mapping.ActionMap()
    conversations = []
    instruction = rf"craft a crafting table"
    image_path = rf"/nfs-shared/pretrain-jarvis-data/trajectory/09-28-craft-crafting_table/image/0cd10446-ddf1-42e5-abfe-0f58a55d459d0.jpg"
    
    conversations.append(processor_wrapper.create_message(prompt=instruction))
    text_prompt = processor_wrapper.create_text_input(conversations=conversations)
    image = processor_wrapper.create_image_input(image_path=image_path)
    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)
    generate_ids = model.to(device).generate(**inputs, max_length=1024,temperature=0.,do_sample=False)
    outputs = processor.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
    action_token = outputs
    
    action = action_map.map(action_token)
    from rich import print
    print(f"text_prompt:{text_prompt}")
    import ipdb 
    ipdb.set_trace()
    #printk(f"outputs:{outputs}")
    #print(f"action_token:{action_token}")
    #print("action:", action)