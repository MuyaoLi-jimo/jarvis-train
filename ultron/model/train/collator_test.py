from transformers import AutoTokenizer, AutoProcessor,AutoModelForCausalLM,GenerationConfig,LlavaNextProcessor
from PIL import Image
import pathlib
import numpy as np
import torch
from torchvision import transforms
from rich.console import Console
from rich import console
from ultron.model.train.utils import prepare_conversation_text_with_images, prepare_conversation_for_molmo,print_trainable_parameters,pad_sequence,transform_image


class ChatDataCollator:
    def __init__(self, tokenizer, max_seq_length = 1024):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
    
    def __call__(self, examples):
        texts = []

        for example in examples:
            # if len(example["images"]) > 1:
            #     raise ValueError("This collator only supports one image per example")
            messages = example["conversations"]
            processed_messages = []
            for message in messages:
                processed_message_role = message["role"]
                processed_message_content = ""
                for item in message["content"]:
                    if item["type"] == "text":
                        processed_message_content += item["text"]
                    else:
                        print("Only text is supported for now.")
                processed_messages.append({"role": processed_message_role, "content": processed_message_content})
            
            text = self.tokenizer.apply_chat_template(
                processed_messages, tokenize=False, add_generation_prompt=False
            )
            # print("text: ", text)
            texts.append(text)

        batch = self.tokenizer(texts, return_tensors="pt", padding="max_length", max_length=self.max_seq_length, truncation=True)

        labels = batch["input_ids"].clone()
        # TODO: add back 
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        return batch

class TextChatDataCollatorForVLM:
    def __init__(self, processor, max_seq_length = 1024):
        self.processor = processor
        self.max_seq_length = max_seq_length
    
    def __call__(self, examples):
        texts = []

        for example in examples:
            if 'text' in example.keys():
                text = example['text']
            elif 'conversations' in example.keys():
                # text = prepare_conversation_text(example, self.processor.tokenizer)
                text = processor.apply_chat_template(example['conversations'], add_generation_prompt=False)
            else:
                print('No text or conversations found in example')
                text = ''
            # print(text)
                # continue
            # print(text)
            texts.append(text)

        batch = self.processor(text = texts, 
                               images = None, 
                               return_tensors="pt", 
                               padding='max_length', 
                               max_length=self.max_seq_length, 
                               truncation=True)
        # import pdb; pdb.set_trace()
        # DONE: mask the tokens from user, only set assistant tokens as attention_mask=1
        # process attention mask
        labels = batch["input_ids"].clone()
        # DONE: add back 
        if self.processor.tokenizer.pad_token_id is not None:
            pad_mask = labels == self.processor.tokenizer.pad_token_id
            if self.processor.tokenizer.pad_token_id == self.processor.tokenizer.eos_token_id:
                pad_mask[:,1:]  = pad_mask[:,1:] & pad_mask[:,:-1]
            labels[pad_mask] = -100
        
        for i, example in enumerate(examples):
            if 'conversations' in example.keys():
                # compute the token ids from user input:
                c_idx = 0
                conv = []
                for c in example['conversations']:
                    conv.append(c)
                    c_text = processor.apply_chat_template(conv, add_generation_prompt=False)
                    c_batch = self.processor(text = [c_text], images = None, return_tensors="pt")
                    token_len = len(c_batch['input_ids'][0])-c_idx
                    if c['role'] == 'user':
                        if c_idx+token_len > self.max_seq_length:
                            # Update: fix attention mask bugs, which will make the lm not see user prompt
                            # batch['attention_mask'][i][c_idx:self.max_seq_length] = 0
                            labels[i][c_idx:self.max_seq_length] = -100 
                        else:
                            # batch['attention_mask'][i][c_idx:c_idx+token_len] = 0
                            labels[i][c_idx:c_idx+token_len] = -100 
                    else: # role == 'assistant'
                        continue
                    c_idx += token_len
                    if c_idx >= self.max_seq_length:
                        break
            elif 'text' in example.keys():
                # batch['attention_mask'][i][:] = 1
                continue
            else:
                # batch['attention_mask'][i][:] = 0
                print("no avaiable text found in this data example")

        batch["labels"] = labels
        return batch

class PlainTextDataCollator:
    def __init__(self, tokenizer, max_seq_length = 1024):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
    
    def __call__(self, examples):
        texts = []

        for example in examples:
            # if len(example["images"]) > 1:
            #     raise ValueError("This collator only supports one image per example")
            # messages = example["conversations"]
            # processed_messages = []
            # for message in messages:
            #     processed_message_role = message["role"]
            #     processed_message_content = ""
            #     for item in message["content"]:
            #         if item["type"] == "text":
            #             processed_message_content += item["text"]
            #         else:
            #             print("Only text is supported for now.")
            #     processed_messages.append({"role": processed_message_role, "content": processed_message_content})
            
            # text = self.tokenizer.apply_chat_template(
            #     processed_messages, tokenize=False, add_generation_prompt=False
            # )
            text = example["text"]
            # print("text: ", text)
            texts.append(text)

        batch = self.tokenizer(texts, return_tensors="pt", padding="max_length", max_length=self.max_seq_length, truncation=True)

        labels = batch["input_ids"].clone()
        # TODO: add back 
        if self.tokenizer.pad_token_id is not None:
            pad_mask = labels == self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                first_pad_indices = None
                if self.tokenizer.padding_side == "right":
                    first_pad_indices = pad_mask.int().argmin(dim=1)
                else:
                    first_pad_indices = pad_mask.int().argmax(dim=1)
                for i, first_pad_idx in enumerate(first_pad_indices):
                    pad_mask[i, first_pad_idx] = False  # 排除第一个 pad_token 的位置
            labels[pad_mask] = -100
        batch["labels"] = labels
        return batch
    
examples = [
    {
        "id": "a3433af3-2b65-401f-8989-b8a4df267209_390",
        "task_id": "6d22ee11-1130-413d-b9a1-2febdcde58a2",
        "label": [
            "trajectory",
            "RT2",
            "craft item crafting table",
            "h=1"
        ],
        "conversations": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Construct a crafting table. \nArrange the materials in the crafting grid according to the following pattern: \n# #\n# #\nEach # represents a plank.\n\n thought: Build a crafting table.\n observation: \n"
                    },
                    {
                        "type": "text",
                        "text": "\n action: <|reserved_special_token_178|><|reserved_special_token_227|><|reserved_special_token_240|><|reserved_special_token_179|>\n thought: Construct a crafting table. \n observation: \n"
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "<|reserved_special_token_178|><|reserved_special_token_227|><|reserved_special_token_239|><|reserved_special_token_179|>\n"
                    }
                ]
            }
        ],
    },
    {
        "image": [
            "image/d4cdc29d-2113-433d-8302-2e272bc3805a_76.jpg",
            "image/d4cdc29d-2113-433d-8302-2e272bc3805a_77.jpg"
        ],
        "conversations": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Make a crafting table.\nArrange the materials in the crafting grid according to the following pattern: \n# #\n# #\nEach # represents a plank.\n\n thought: Construct a bucket by utilizing the crafting table in Minecraft.\n observation: \n"
                    },
                    {
                        "type": "image",
                        "text": "<image>"
                    },
                    {
                        "type": "text",
                        "text": "\n action: <|reserved_special_token_178|><|reserved_special_token_211|><|reserved_special_token_248|><|reserved_special_token_179|>\n thought: Make a crafting table.\n observation: \n"
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
                        "text": "<|reserved_special_token_178|><|reserved_special_token_211|><|reserved_special_token_248|><|reserved_special_token_179|>\n"
                    }
                ]
            }
        ],
    }
]
    
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
            self.user_template = "USER: "
            self.assistant_template = "ASSISTANT: "
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
            assert len(beg_matches)==len(end_matches)
            label[:beg_matches[0]]=-100
            for instruction_beg_idx,instruction_end_idx in zip(beg_matches,end_matches):
                label[instruction_beg_idx:instruction_end_idx]= -100
            

        if self.processor.tokenizer.pad_token_id is not None:
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        #import pdb; pdb.set_trace()
        return batch
    
if __name__ == "__main__":
    

    
    processor_config = dict(
        do_rescale=False,
        patch_size=14,
        vision_feature_select_strategy="default"
    )
    processor = LlavaNextProcessor.from_pretrained("/scratch/mc_lmy/models/llama3-llava-next-8b-hf",**processor_config)
    #if getattr(processor.tokenizer, "pad_token", None) is None:
    processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "right"
    """ 
    processor = AutoProcessor.from_pretrained(
        "/scratch/models/molmo-7b-d-0924",
        trust_remote_code=True,
        torch_dtype='auto',
        device_map='auto'
    )
    """
    #torch.set_printoptions(threshold=10000)
    data_collator = TextChatDataCollatorForVLM(processor, max_seq_length = 2048)
    torch.set_printoptions(threshold=10000)
    #data_collator = MultimodalDataCollator(processor, image_folder="/scratch/mc_lmy/datas/11-10-craft-craft_table-shell_agent-hard",max_seq_length = 4096,model_name_or_path="/scratch/mc_lmy/models/llama3-llava-next-8b-hf")
    output= data_collator(examples)
    new_labels = output["labels"][0][output["labels"][0] != -100]
    print(new_labels)
    print(processor.tokenizer.convert_ids_to_tokens(new_labels))
    print(processor.tokenizer.convert_ids_to_tokens(output["input_ids"][0]))
    #print(output)
    is_target = (output["input_ids"][0] == 128256)

    # 找到布尔张量中值变化的位置
    # `1` 表示开始，`-1` 表示结束
    diff = torch.diff(is_target.int(), prepend=torch.tensor([0]))

    # 找到开始和结束的位置
    start_positions = (diff == 1).nonzero(as_tuple=True)[0].tolist()
    end_positions = (diff == -1).nonzero(as_tuple=True)[0].tolist()

    # 输出结果
    print("Start positions:", start_positions)
    print("End positions:", end_positions)
    
    #print(output["labels"])

    exit()
    from datasets import load_dataset
    dataset_name="/home/limuyao/datas/jarvis-dataset-004/11-06-craft-craft_table-shell_agent-normal-mistral"
        
    train_dataset_file = dataset_name + "-train.json"
    eval_dataset_file = dataset_name + "-valid.json"

    raw_datasets = load_dataset("json", data_files={"train": train_dataset_file, "validation": eval_dataset_file}, num_proc=8)

    train_dataset = raw_datasets['train']
    eval_dataset = raw_datasets['validation']
    train_dataset = train_dataset.shuffle(27)
    print(train_dataset[0])
    exit()
    
    
    
    