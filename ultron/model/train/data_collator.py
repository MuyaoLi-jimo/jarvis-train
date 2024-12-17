import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import pathlib
from rich import console
from ultron.model.train.utils import (
    prepare_conversation_text_with_images,
    prepare_conversation_for_molmo,
    pad_sequence,
    transform_image,
)
from transformers import AutoProcessor
################
# Create a data collator to encode text and image pairs
################

class MultimodalDataCollator:
    def __init__(self, processor:AutoProcessor,model_name_or_path, image_folder = '/nfs-shared/data/JARVIS/tmp/images', with_image = True, resize_image = True, max_seq_length = 1024,check:bool=False):
        self.processor = processor
        self.model_type = None
        self.user_template = None
        self.assistant_template = None
        self.tokenize_redundant = 0
        model_name_or_path = model_name_or_path.lower()
        self.model_name_or_path = model_name_or_path
        if "molmo" in model_name_or_path:
            self.model_type = "molmo"
            self.user_template = " User:"
            self.assistant_template = " Assistant:"
        elif "mistral" in model_name_or_path:
            self.model_type = "mistral"
            self.user_template = "[INST]"
            self.assistant_template = "[/INST]"
            self.tokenize_redundant = 1
        elif 'vicuna' in model_name_or_path:
            self.model_type = "llama-2"
            self.user_template = " USER: "
            self.assistant_template = " ASSISTANT: "
            self.tokenize_redundant = 1
        elif "llama-3" in model_name_or_path or "llama3" in  model_name_or_path or "llama_3" in model_name_or_path:
            self.model_type = "llama-3"
            self.user_template ="<|start_header_id|>user<|end_header_id|>"
            self.assistant_template = "<|start_header_id|>assistant<|end_header_id|>"
            self.tokenize_redundant = 1
        self.image_folder = image_folder
        self.with_image = with_image
        self.resize_image = resize_image
        self.random_image_width = 224
        self.random_image_height = 224
        self.default_image_size = (672,336) # with this image size, the llava-next will split it into 3 patches, not 5 pathces in 640*360
        self.max_seq_length = max_seq_length
        self.check = check
        self.my_console = console.Console()

            
    
    def __call__(self, examples):
        texts = []
        if self.with_image:
            images = []
        else:
            images = None

        for example in examples:
            if 'text' in example.keys():
                text = example['text']
            elif 'conversations' in example.keys():  
                if "molmo" in self.model_name_or_path:
                    text = prepare_conversation_for_molmo(example)
                elif "vicuna" in self.model_name_or_path:
                    text = self.processor.tokenizer.apply_chat_template(example["conversations"],tokenize=False,)
                else:
                    text = prepare_conversation_text_with_images(example, self.processor.tokenizer)  #合并<image>，并转化为加入chat template的版本
            else:
                print('No text or conversations found in example')
                text = ''
                # continue
            texts.append(text)
            # 处理图片(如果存在的话)
            if self.with_image and example.get('image'):
                image_paths = example.get('image')
                if isinstance(image_paths, list):
                    pass
                elif isinstance(image_paths, str):
                    image_paths = [image_paths]
                else:
                    raise ValueError("example_image must be a string or a list of strings.")
                image_num = len(image_paths)
                for idx,image_path in enumerate(image_paths):
                    if image_path[0]!="/": #if not abs path
                        image_path = pathlib.Path(self.image_folder)/image_path
                    else:
                        image_path = pathlib.Path(image_path)
                    
                    if not image_path.exists():
                        raise ValueError(f"Image file {image_path} not found.")
                    else:
                        image = Image.open(image_path)
                        image=transform_image(image)
                        if "llava" in self.model_name_or_path and self.resize_image:
                            image = image.resize(self.default_image_size)
                            # 创建一个 transform 对象，将 PIL.Image 转换为 Tensor
                        if "molmo" not in self.model_name_or_path:
                            transform = transforms.ToTensor()
                            # 将图像转换为 Tensor
                            image = transform(image)
                        
                        images.append(image)
        if len(images) == 0:
            images = None
            
        if self.check:#检查长度
            batch_input_ids = self.processor(text = texts, images = images,)["input_ids"]
            batch_length_dict = {e["id"]:len(batch_input_id) for batch_input_id,e in zip(batch_input_ids,examples)}
            return batch_length_dict

        #prepare the batches
        if self.model_type =="molmo":  #truncation=True
            image_idx = 0
            batch_inputs = []
            batch = {}
            for user_text,assistant_text,image_num in texts:
                inputs = self.processor.process(
                    images=images[image_idx:image_idx+image_num],
                    text=user_text
                )
                tokens = self.processor.tokenizer.encode(assistant_text, add_special_tokens=False)
                tokens_tensor = torch.tensor(tokens, dtype=torch.long)
                eos_tensor = torch.tensor([self.processor.tokenizer.eos_token_id], dtype=torch.long)
                inputs["input_ids"] = torch.cat([inputs["input_ids"], tokens_tensor, eos_tensor])
                batch_inputs.append(inputs)
                image_idx += image_num

            input_ids = [b['input_ids'].clone().detach() for b in batch_inputs]
            batch['input_ids'] = pad_sequence(input_ids, padding_value=self.processor.tokenizer.pad_token_id, max_length=self.max_seq_length, truncation=True)
            # Stack other elements
            for key in batch_inputs[0]:
                if key != 'input_ids':
                    batch[key] = torch.stack([b[key].clone().detach() for b in batch_inputs], dim=0)
                    
        else:
            batch = self.processor(text = texts, images = images, return_tensors="pt", padding='max_length', max_length=self.max_seq_length, truncation=True)
       #torch.set_printoptions(threshold=10000)
        labels = batch["input_ids"].clone()
        check_id = -1 if self.processor.tokenizer.padding_side=="right" else 0
        if labels[0][check_id].item()!=self.processor.tokenizer.pad_token_id:
            self.my_console.log("[red]Warning! the token length is probably out of max token length!")
        # TODO: add back -- 非常重要
        
        for label in labels:
            np_label = label.cpu().numpy()
            instruction_beg_token_ids =  np.array(self.processor.tokenizer(self.user_template).input_ids[self.tokenize_redundant:]) #remove <s>
            instruction_end_token_ids = np.array(self.processor.tokenizer(self.assistant_template).input_ids[self.tokenize_redundant:]) #remove <s>
            label_len,beg_len,end_len = len(label), len(instruction_beg_token_ids), len(instruction_end_token_ids)
            beg_matches = np.where((np_label[np.arange(label_len - beg_len + 1)[:, None] + np.arange(beg_len)] == instruction_beg_token_ids).all(axis=1))[0].tolist()
            end_matches = np.where((np_label[np.arange(label_len - end_len + 1)[:, None] + np.arange(end_len)] == instruction_end_token_ids).all(axis=1))[0].tolist()
            len_beg_matches = len(beg_matches)
            len_end_matches = len(end_matches)
            if len_beg_matches==len_end_matches+1:
                end_matches.append(self.max_seq_length)
                len_end_matches +=1
                self.my_console.log("[red]Warning! the token length is probably out of max token length!")
            assert len_beg_matches==len_end_matches
            label[:beg_matches[0]]=-100
            for instruction_beg_idx,instruction_end_idx in zip(beg_matches,end_matches):
                label[instruction_beg_idx:instruction_end_idx]= -100
            
        if self.processor.tokenizer.pad_token_id is not None:
            pad_mask = labels == self.processor.tokenizer.pad_token_id
            if self.processor.tokenizer.pad_token_id == self.processor.tokenizer.eos_token_id:
                pad_mask[:,1:]  = pad_mask[:,1:] & pad_mask[:,:-1]
            labels[pad_mask] = -100
        #import pdb; pdb.set_trace() 
        batch["labels"] = labels
        
        return batch
    
if __name__ == "__main__":
    processor_config = dict(
        do_rescale=False,
        patch_size=14,
        vision_feature_select_strategy="default"
    )
    from transformers import LlavaNextProcessor
    #processor = LlavaNextProcessor.from_pretrained("/scratch/mc_lmy/models/llama3-llava-next-8b-hf",**processor_config)
    processor = LlavaNextProcessor.from_pretrained("/scratch/mc_lmy/models/llava-v1.6-vicuna-7b-hf",**processor_config)
    VICUNA_CHAT_TEMPLATE = """{{ bos_token }}{% for message in messages %}{% if message['role'] != 'system' %}{{ ' '+message['role'].upper() + ': '}}{% endif %}{# Render all images first #}{% for content in message['content'] | selectattr('type', 'equalto', 'image') %}{{ '<image>\n' }}{% endfor %}{# Render all text next #}{% if message['role'] != 'assistant' %}{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}{{ content['text'] + '\n'}}{% endfor %}{% else %}{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}{% generation %}{{ content['text'] + eos_token + '\n' }}{% endgeneration %}{% endfor %}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'ASSISTANT:' }}{% endif %}"""
    processor.tokenizer.chat_template = VICUNA_CHAT_TEMPLATE
    """ 
    processor = AutoProcessor.from_pretrained(
        "/scratch/models/molmo-7b-d-0924",
        trust_remote_code=True,
        torch_dtype='auto',
        device_map='auto'
    )
    """
    processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "right"
    examples = [{'id': '910fef70-598c-4904-ac6a-acb25cecc2ed_176', 
          'label': ['trajectory', 'RT2', 'craft item crafting table', 'h=0', 'c=1'], 'image': ['image/910fef70-598c-4904-ac6a-acb25cecc2ed_176.jpg'], 'conversations': [{'content': [{'text': 'Let’s craft a crafting table for this project.\nArrange the materials in the crafting grid according to the following pattern: \n\n plank | plank \n plank | plank \n\nthought: craft a crafting table.\nobservation: \n', 'type': 'text'}, {'text': '<image>', 'type': 'image'}], 'role': 'user'}, {'content': [{'text': '유라요', 'type': 'text'}], 'role': 'assistant'}, {'content': [{'text': '유라요', 'type': 'text'}], 'role': 'user'},{'content': [{'text': '유라요', 'type': 'text'}], 'role': 'assistant'}], 'action': '유라요'}                                                                                       
    ] 
    torch.set_printoptions(threshold=10000)
    data_collator = MultimodalDataCollator(processor, image_folder="/scratch/mc_lmy/datas/11-10-craft-craft_table-shell_agent-hard",max_seq_length = 4096,model_name_or_path="/scratch/mc_lmy/models/llava-v1.6-vicuna-7b-hf")
    output= data_collator(examples)
    #print(output)