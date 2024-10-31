from pathlib import Path
import pathlib
from typing import Union
import base64
from io import BytesIO
import numpy as np
import ultron.model.inference.load_model as load_model
import ultron.model.inference.action_mapping as action_mapping
from PIL import Image
import cv2

def pil2base64(image):
    """强制中间结果为jpeg""" 
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str



class ProcessorWrapper:
    def __init__(self,processor=None,model_name= "llava-next"):
        self.processor = processor
        self.model_name = model_name
        self.default_image_size = (672,336)

    def create_message_vllm(self,image,role="user",input_type="image",prompt:str=""):
        if input_type=="image":
            message = {
                "role":role,
                "content": [
                    {
                    "type": "text",
                    "text": f"{prompt}"
                    },
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{pil2base64(image)}"
                    },
                    },
                ]
            }
        return message


    def create_message(self,role="user",input_type="image",prompt:str="",image=None):

        if self.model_name=="llava-next":
            if input_type=="image":
                message = {
                    "role":role,
                    "content":[
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ]
                }
            if input_type == "text":
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
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image.astype('uint8'))
            image = image.resize(self.default_image_size)
        return image

    

if __name__ == "__main__":


    #exit()
    device = "cuda:4"
    processor,model,LLM_backbone,VLM_backbone = load_model.load_visual_model(checkpoint_path=r"/home/mc_lmy/model/mc-llava_v1.6_vicuna_mistral_7b-LORA-embodied_mini_craft_10-10-08-llava-v1.6-A100-c4-e3-b16-a4")
    processor_wrapper = ProcessorWrapper(processor,model_name=VLM_backbone)
    action_map = action_mapping.ActionMap(tokenizer_type=LLM_backbone)
    conversations = []
    instruction = rf"craft a crafting table"
    image_path = rf"/home/mc_lmy/workspace/jarvis-train/test1.jpg"
    
    conversations.append(processor_wrapper.create_message(prompt=instruction))
    text_prompt = processor_wrapper.create_text_input(conversations=conversations)
    text_prompt+= "ಮ"
    image = processor_wrapper.create_image_input(image_path=image_path)
    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)
    generate_ids = model.to(device).generate(**inputs, max_length=1024,temperature=0.5)

    from rich import print
    
    outputs = processor.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
    print(outputs)
    #action_token = outputs
    action = action_map.map(outputs)

    print(f"action:{action}")
