from transformers import AutoTokenizer, AutoProcessor,AutoModelForCausalLM,GenerationConfig,LlavaNextProcessor
from PIL import Image
import pathlib
import numpy as np
import torch
from torchvision import transforms
from ultron.model.train.utils import prepare_conversation_text_with_images, prepare_conversation_for_molmo,print_trainable_parameters,pad_sequence

class MultimodalDataCollator:
    def __init__(self, processor,model_name_or_path, image_folder = '/nfs-shared/data/JARVIS/tmp/images', with_image = True, resize_image = True, max_seq_length = 1024):
        self.processor = processor
        self.model_name_or_path = model_name_or_path
        self.image_folder = image_folder
        self.with_image = with_image
        self.resize_image = resize_image
        self.random_image_width = 224
        self.random_image_height = 224
        self.default_image_size = (672,336) # with this image size, the llava-next will split it into 3 patches, not 5 pathces in 640*360
        self.max_seq_length = max_seq_length
        self.no_image_policy = 'random' # 'random' or 'ignore'
        self.user_template = None
        self.assistant_template = None
        self.tokenize_redundant = 0
        if "llava" in self.model_name_or_path:
            self.user_template = "[INST]"
            self.assistant_template = "[/INST]"
            self.tokenize_redundant = 1
        elif "molmo" in self.model_name_or_path:
            self.user_template = " User:"
            self.assistant_template = " Assistant:"
    
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
                if "llava" in self.model_name_or_path:
                    text = prepare_conversation_text_with_images(example, self.processor.tokenizer)  #合并<image>，并转化为加入chat template的版本
                elif "molmo" in self.model_name_or_path:
                    text = prepare_conversation_for_molmo(example)
            else:
                print('No text or conversations found in example')
                text = ''
                # continue
            texts.append(text)
            # 处理图片
            if example['image'] and self.with_image:
                if isinstance(example['image'], list):
                    image_paths = example['image']
                elif isinstance(example['image'], str):
                    image_paths = [example['image']]
                else:
                    raise ValueError("example_image must be a string or a list of strings.")
                
                for image_path in image_paths:
                    if image_path[0]!="/": #if not abs path
                        image_path = pathlib.Path(self.image_folder)/image_path
                    else:
                        image_path = pathlib.Path(image_path)
                    
                    if not image_path.exists():
                        print(f"Image file {image_path} not found, set a random image instead.")
                        random_image_dim = [self.random_image_width, self.random_image_height]
                        image = torch.rand(3, *random_image_dim)
                        images.append(image)
                    else:
                        image = Image.open(image_path)
                        
                        if "llava" in self.model_name_or_path and self.resize_image:
                            image = image.resize(self.default_image_size)
                            # 创建一个 transform 对象，将 PIL.Image 转换为 Tensor
                        if "molmo" not in self.model_name_or_path:
                            transform = transforms.ToTensor()
                            # 将图像转换为 Tensor
                            image = transform(image)
                        images.append(image)

        if self.with_image and len(images) == 0:
            images = None
            
        #prepare the batches
        if "molmo" in self.model_name_or_path:  #truncation=True
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
        
       #
        labels = batch["input_ids"].clone()
        # TODO: add back -- 非常重要
        for label in labels:
            np_label = np.array(label)
            cur_len = 0
            instruction_beg_token_ids =  np.array(self.processor.tokenizer(self.user_template).input_ids[self.tokenize_redundant:]) #remove <s>
            instruction_end_token_ids = np.array(self.processor.tokenizer(self.assistant_template).input_ids[self.tokenize_redundant:]) #remove <s>
            """ 
            if self.processor.tokenizer.padding_side == "left":
                padding_len = sum(label==self.processor.tokenizer.pad_token_id)
                cur_len = padding_len + 2 #tokenizer：1<s>，chat_template:1<s>
            else:
                cur_len = 1
            label[0:cur_len] = -100
            """
            label_len,beg_len,end_len = len(label), len(instruction_beg_token_ids), len(instruction_end_token_ids)
            beg_matches = np.where((np_label[np.arange(label_len - beg_len + 1)[:, None] + np.arange(beg_len)] == instruction_beg_token_ids).all(axis=1))[0].tolist()
            end_matches = np.where((np_label[np.arange(label_len - end_len + 1)[:, None] + np.arange(end_len)] == instruction_end_token_ids).all(axis=1))[0].tolist()
            assert len(beg_matches)==len(end_matches)
            label[:beg_matches[0]]=-100
            for instruction_beg_idx,instruction_end_idx in zip(beg_matches,end_matches):
                label[instruction_beg_idx:instruction_end_idx]= -100
            

        if self.processor.tokenizer.pad_token_id is not None:
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        return batch
    
examples = [
    {
        "id": "ad1a0cdb-93d1-4515-9a26-8376489e569528",
        "image": "image/ad1a0cdb-93d1-4515-9a26-8376489e569528.jpg",
        "conversations": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Construct a crafting table."
                    },
                    {
                        "type": "image",
                        "text": "<image>"
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "\u0cae\u5c97\u57f7\u12a0"
                    }
                ]
            }
        ]
    },
    {
        "id": "3d966b41-2299-4acd-b4b1-3fbbd7e653e4580",
        "image": "image/3d966b41-2299-4acd-b4b1-3fbbd7e653e4580.jpg",
        "conversations": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Construct a crafting table."
                    },
                    {
                        "type": "image",
                        "text": "<image>"
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "\u0cae\u5be6\u5c97\ucee8\u12a0"
                    }
                ]
            }
        ],
    },
]
    
if __name__ == "__main__":
    
    processor_config = dict(
        do_rescale=False,
        patch_size=14,
        vision_feature_select_strategy="default"
    )
    processor = LlavaNextProcessor.from_pretrained("/home/mc_lmy/model/llava-v1.6-mistral-7b-hf",**processor_config)
    """ 
    processor = AutoProcessor.from_pretrained(
        "/scratch/models/molmo-7b-d-0924",
        trust_remote_code=True,
        torch_dtype='auto',
        device_map='auto'
    )
    """
    torch.set_printoptions(threshold=10000)
    data_collator = MultimodalDataCollator(processor, image_folder="/home/mc_lmy/datas/jarvis-dataset-003",max_seq_length = 2048,model_name_or_path="/home/mc_lmy/model/llava-v1.6-mistral-7b-hf")
    output= data_collator(examples)
    print(output["labels"])
