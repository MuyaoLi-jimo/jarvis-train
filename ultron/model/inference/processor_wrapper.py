from pathlib import Path
import pathlib
from typing import Union,Literal
import base64
from io import BytesIO
import numpy as np
import ultron.model.inference.load_model as load_model
#import ultron.model.inference.action_mapping as action_mapping
from PIL import Image
import cv2
import torch
import requests
import io
import re

def pil2base64(image):
    """强制中间结果为jpeg""" 
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str


def encode_image_to_base64(image:Union[str,pathlib.PosixPath,Image.Image,np.ndarray], format='JPEG') -> str:
    """Encode an image to base64 format, supports URL, numpy array, and PIL.Image."""

    # Case 1: If the input is a URL (str)
    image_encode = None
    if isinstance(image, str) and image[:4]=="http":
        try:
            response = requests.get(image)
            response.raise_for_status()
            return base64.b64encode(response.content).decode('utf-8')
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Failed to retrieve the image from the URL: {e}")
    elif isinstance(image, str) and image[0]=='/':
        with image.open('rb') as image_file:
            image_encode =  base64.b64encode(image_file.read()).decode('utf-8')
        return image_encode
    elif isinstance(image,pathlib.PosixPath):
        with image.open('rb') as image_file:
            image_encode =  base64.b64encode(image_file.read()).decode('utf-8')
        return image_encode
    # Case 3: If the input is a numpy array
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        buffer.seek(0)
        image_bytes = buffer.read()
        return base64.b64encode(image_bytes).decode('utf-8')

    # Case 4: If the input is a PIL.Image
    elif isinstance(image, Image.Image):
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        buffer.seek(0)
        image_bytes = buffer.read()
        return base64.b64encode(image_bytes).decode('utf-8')

    # Raise an error if the input type is unsupported
    else:
        raise ValueError("Unsupported input type. Must be a URL (str), numpy array, or PIL.Image.")

def get_suffix(image:Union[list,str,pathlib.PosixPath,np.ndarray,Image.Image]):
    if isinstance(image,np.ndarray|Image.Image):
        image_suffix = "JEPG" 
    elif isinstance(image,str):
        image_suffix = image.split(".")[-1]
    elif isinstance(image,pathlib.PosixPath):
        image_suffix = image.suffix[1:]
    else:
        raise ValueError(f"invalid image type！")
    return image_suffix

def translate_cv2(image: Union[str, pathlib.PosixPath, np.ndarray, Image.Image]) -> np.ndarray:
    if isinstance(image, Image.Image):
        # Convert PIL Image to NumPy array (PIL is in RGB)
        img_array = np.array(image)
        cv2_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    elif isinstance(image, np.ndarray):
        # Check if the NumPy array is in RGB format and has three channels
        if image.shape[2] == 3:  # Only for color images
            cv2_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            cv2_image = image  # No conversion needed for grayscale images
    elif isinstance(image, (str, pathlib.PosixPath)):
        # Read the image using cv2 (assumes BGR format)
        cv2_image = cv2.imread(str(image))  # Convert PosixPath to string if necessary
        if cv2_image is None:
            raise ValueError(f"The image path is incorrect or the file is not accessible: {image}")
    else:
        raise ValueError("Unsupported image format or path type")
    
    return cv2_image

    

class ProcessorWrapper:
    def __init__(self,processor=None,model_name= "llava-next"):
        self.processor = processor
        self.model_name = model_name
        self.image_size_map = {
            "llava_next":(672,336),
            "molmo":(2560,1440),
        }

    def create_message_vllm(self,image:Union[list,str,pathlib.PosixPath,np.ndarray,Image.Image],role:Literal["user","assistant"]="user",input_type:Literal["image","text"]="image",prompt:str=""):
        if role not in {"user","assistant"}:
            raise ValueError(f"a invalid role {role}")
        
        if input_type=="image":
            image = encode_image_to_base64(image)
            image_suffix = get_suffix(image)
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
                        "url": f"data:image/{image_suffix};base64,{image}"
                    },
                    },
                ]
            }
        return message

    def create_message(self,role="user",input_type="image",prompt:str="",image=None):

        if self.model_name in {"llava-next","molmo"} :
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
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image.astype('uint8'))
        if self.model_name in {"llava-next","molmo"}:
            image = image.resize(self.image_size_map[self.model_name])
        return image

class MolmoOutProcess(object):
    def __init__(self,):
        self.point_pattern = r'<point x="([0-9]+\.?[0-9]*)"\s+y="([0-9]+\.?[0-9]*)"\s+alt="([^"]+)">[^<]+</point>'
        self.points_pattern = r'<points(.*?)\salt="([^"]*)">[^<]+</points>'
        self.x_pattern = r'x[0-9]+="([0-9]+\.?[0-9]*)"'
        self.y_pattern = r'y[0-9]+="([0-9]+\.?[0-9]*)"'
        
    def parse_point(self,input_str):
        match = re.search(self.point_pattern, input_str)
        if match:
            x_percentage = float(match.group(1)) / 100
            y_percentage = float(match.group(2)) / 100
            label = match.group(3)
            return x_percentage, y_percentage, label
        else:
            return None

    def parse_points(self,input_str):
        match = re.search(self.points_pattern, input_str)
        if match:
            results = []
            coords_text = match.group(1)
            label = match.group(2)
            x_values = re.findall(self.x_pattern, coords_text)
            y_values = re.findall(self.y_pattern, coords_text)
            # 检查坐标点是否配对完整
            for x, y in zip(x_values, y_values):
                x_percentage = float(x) / 100
                y_percentage = float(y) / 100
                results.append((x_percentage, y_percentage, label))
            return results
        else:
            return None
    
    def parse(self,input_str):
        result = self.parse_point(input_str)
        if result:
            return [result]
        results = self.parse_points(input_str)
        return results
    
    def point_with_guide(self,image:Union[str,pathlib.PosixPath,np.ndarray,Image.Image],guides:list)->np.ndarray:
        cv2_image = translate_cv2(image)
        if not guides:
            # 如果guides是None
            return cv2_image
    
        height, width = cv2_image.shape[:2]
        # Set properties for drawing
        color = (255, 105, 180)  # Pink color in BGR format
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 2
        radius = 4  # Radius of the circle to draw
        for guide in guides:
            x_percent, y_percent, text = guide[0], guide[1], guide[2]
        
            # Calculate actual coordinates from percentages
            x_coord = int(x_percent * width)
            y_coord = int(y_percent * height)

            # Draw a large pink point on the image
            cv2.circle(cv2_image, (x_coord, y_coord), radius, color, -1)  # -1 fill the circle

            # Put the text near the point, slightly to the right
            cv2.putText(cv2_image, text, (x_coord + 15, y_coord), font, font_scale, color, thickness)

        return cv2_image
    
    def point(self,image:Union[str,pathlib.PosixPath,np.ndarray,Image.Image],input_str:str):
        guides = self.parse(input_str)
        return self.point_with_guide(image,guides)

    
if __name__ == "__main__":

    #exit()
    device = "cuda:6"
    processor,model,LLM_backbone,VLM_backbone = load_model.load_visual_model(device=device,checkpoint_path=r"/scratch/mc_lmy/models/mc-llava_v1.6_mistral_7b-LORA-embodied_mini_10-30-craft-craft_table-shell_agent-normal-mistral-10-30-A100-c4-e10-b16-a1-576")
    processor_wrapper = ProcessorWrapper(processor,model_name=VLM_backbone)
    image_paths = [rf"/home/mc_lmy/datas/jarvis-dataset-003/image/3d966b41-2299-4acd-b4b1-3fbbd7e653e491.jpg",
                   rf"/home/mc_lmy/datas/jarvis-dataset-003/image/3d966b41-2299-4acd-b4b1-3fbbd7e653e491.jpg",
                  rf"/home/mc_lmy/datas/jarvis-dataset-003/image/3d966b41-2299-4acd-b4b1-3fbbd7e653e492.jpg",
                  rf"/home/mc_lmy/datas/jarvis-dataset-003/image/3d966b41-2299-4acd-b4b1-3fbbd7e653e493.jpg",
                  ]

    from rich import print
    conversations=[]
    conversations.append(processor_wrapper.create_message(prompt="."))
    text_prompt = processor_wrapper.create_text_input(conversations=conversations)
    past_image_tokens = None
    for i,image_path in enumerate(image_paths):
        image = processor_wrapper.create_image_input(image_path=image_path)
        inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)
        #print(inputs)
        #print(text_prompt)
        image_token = model.get_image_features(pixel_values=inputs.pixel_values,image_sizes=inputs.image_sizes, vision_feature_layer=-2, vision_feature_select_strategy="default")
        if i>0:
            if torch.equal(past_image_tokens, image_token[0]):
                print("坏了")
            else:
                print(torch.sum(past_image_tokens != image_token[0]).item())
        print(image_token[0].shape)
        past_image_tokens = image_token[0]
