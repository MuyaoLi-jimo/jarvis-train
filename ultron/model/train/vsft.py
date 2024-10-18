import logging
import os
from contextlib import nullcontext
from PIL import Image
import pathlib
import json
import numpy as np
from torchvision import transforms

TRL_USE_RICH = os.environ.get("TRL_USE_RICH", False)

from trl.commands.cli_utils import init_zero_verbose, SFTScriptArguments, TrlParser

if TRL_USE_RICH:
    init_zero_verbose()
    FORMAT = "%(message)s"
    from rich.console import Console
    from rich.logging import RichHandler

import torch
from accelerate import Accelerator
from datasets import load_dataset, Dataset

from tqdm.rich import tqdm
from transformers import AutoTokenizer, AutoProcessor
from transformers import FuyuProcessor, LlavaProcessor, Blip2Processor, LlavaNextProcessor
from transformers import LlavaForConditionalGeneration, FuyuForCausalLM, Blip2ForConditionalGeneration, LlavaNextForConditionalGeneration

from trl import (
    ModelConfig,
    RichProgressCallback,
    SFTConfig,
    SFTTrainer,
    get_peft_config,
    get_quantization_config,
    get_kbit_device_map,
)
from rich import print
from ultron.model.train.utils import prepare_conversation_text, print_trainable_parameters

tqdm.pandas()

if TRL_USE_RICH:
    logging.basicConfig(format=FORMAT, datefmt="[%X]", handlers=[RichHandler()], level=logging.INFO)

################
# Create a data collator to encode text and image pairs
################

class MultimodalDataCollator:
    def __init__(self, processor, image_folder = '/nfs-shared/data/JARVIS/tmp/images', with_image = True, resize_image = True, max_seq_length = 1024):
        self.processor = processor
        self.image_folder = image_folder
        self.with_image = with_image
        self.resize_image = True
        self.random_image_width = 224
        self.random_image_height = 224
        self.default_image_size = (672,336) # with this image size, the llava-next will split it into 3 patches, not 5 pathces in 640*360åœ
        self.max_seq_length = max_seq_length
        self.no_image_policy = 'random' # 'random' or 'ignore'
    
    def __call__(self, examples):
        texts = []
        if self.with_image:
            images = []
        else:
            images = None

        for example in examples:
            # DISCARD -> use unified prepare_conversation_text function ! 
            # messages = example["conversations"]
            # processed_messages = []
            # for message in messages:
            #     processed_message_role = message["role"]
            #     processed_message_content = ""
            #     for item in message["content"]:
            #         if item["type"] == "text":
            #             processed_message_content += item["text"]
            #         elif item["type"] == "image":
            #             processed_message_content += "<image>"
            #     processed_messages.append({"role": processed_message_role, "content": processed_message_content})

            # if not example['image']:
            #     print("No image found in the example, set a random image instead.")
            #     processed_messages[0]['content'] = '<image>\nThe image has nothing in it, ignore the image.\n' + processed_messages[0]['content']

            # text = self.processor.tokenizer.apply_chat_template(
            #     processed_messages, tokenize=False, add_generation_prompt=False
            # )
            if 'text' in example.keys():
                text = example['text']
            elif 'conversations' in example.keys():
                text = prepare_conversation_text(example, self.processor.tokenizer)
            else:
                print('No text or conversations found in example')
                text = ''
                # continue
            texts.append(text)

            # if not self.with_image:
            #     continue

            # if not example['image']:
            #     if self.no_image_policy == 'random':
            #         print("No image found in the example, set a random image instead.")
            #         random_image_dim = [self.random_image_width, self.random_image_height]
            #         image = torch.rand(3, *random_image_dim)
            #         images.append(image)
            # else:
            #     # image_path = os.path.join(self.image_folder, example['image'])
            #     image_path = example['image']
            #     if not os.path.exists(image_path):
            #         print(f"Image file {image_path} not found ..")
            #         if self.no_image_policy == 'random':
            #             print("set a random image instead.")
            #             random_image_dim = [self.random_image_width, self.random_image_height]
            #             image = torch.rand(3, *random_image_dim)
            #             images.append(image)
            #     else:
            #         image = Image.open(image_path)
            #         # 创建一个 transform 对象，将 PIL.Image 转换为 Tensor
            #         transform = transforms.ToTensor()
            #         # 将图像转换为 Tensor
            #         tensor_image = transform(image)
            #         # images.append(image)
            #         images.append(tensor_image)
            if example['image'] and self.with_image:
                if isinstance(example['image'], list):
                    image_paths = example['image']
                elif isinstance(example['image'], str):
                    image_paths = [example['image']]
                else:
                    raise ValueError("example_image must be a string or a list of strings.")
                
                for image_path in image_paths:
                    if image_path[0]!="/":
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
                        if self.resize_image:
                        
                            image = image.resize(self.default_image_size)
                        # 创建一个 transform 对象，将 PIL.Image 转换为 Tensor
                        transform = transforms.ToTensor()
                        # 将图像转换为 Tensor
                        tensor_image = transform(image)
                        images.append(tensor_image)

        if self.with_image and len(images) == 0:
            images = None
        batch = self.processor(text = texts, images = images, return_tensors="pt", padding='max_length', max_length=self.max_seq_length, truncation=True)
        # batch = self.processor(text = texts, images = images, return_tensors="pt", padding=True)
        # import pdb; pdb.set_trace()
        labels = batch["input_ids"].clone()
        # TODO: add back 
        if self.processor.tokenizer.pad_token_id is not None:
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        return batch


if __name__ == "__main__":
    
    parser = TrlParser((SFTScriptArguments, SFTConfig, ModelConfig))
    sft_script_args, training_args, model_config = parser.parse_args_and_config()

    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    # Force use our print callback
    if TRL_USE_RICH:
        training_args.disable_tqdm = True
        console = Console()

    ################
    # Model, Tokenizer & Processor
    ################
    ### discard: if no chat_template is defined in tokenizer_config.json, use the default one
    DEFAULT_CHAT_TEMPLATE = """{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = message['role'] + ':\n\n'+ message['content'] + '\n' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}"""
    VICUNA_CHAT_TEMPLATE = """{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] | trim + '\n\n' %}{% set messages = messages[1:] %}{% else %}{% set system_message = '' %}{% endif %}{{ bos_token + system_message }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ 'USER: ' + message['content'] | trim + '\n' }}{% elif message['role'] == 'assistant' %}{{ 'ASSISTANT: ' + message['content'] | trim + eos_token + '\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'ASSISTANT:' }}{% endif %}"""
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    if 'llava-next' in model_config.model_name_or_path or 'llava-v1.6' in model_config.model_name_or_path:
        processor = LlavaNextProcessor.from_pretrained(model_config.model_name_or_path)
        model = LlavaNextForConditionalGeneration.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    elif 'llava-1.5' in model_config.model_name_or_path or 'llava-gemma' in model_config.model_name_or_path:
        processor = LlavaProcessor.from_pretrained(model_config.model_name_or_path)
        model = LlavaForConditionalGeneration.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    elif 'fuyu' in model_config.model_name_or_path:
        processor = FuyuProcessor.from_pretrained(model_config.model_name_or_path)
        model = FuyuForCausalLM.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    elif 'blip2' in model_config.model_name_or_path:
        processor = Blip2Processor.from_pretrained(model_config.model_name_or_path)
        model = Blip2ForConditionalGeneration.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    else:
        processor = AutoProcessor.from_pretrained(model_config.model_name_or_path)
    
    if not processor.tokenizer.chat_template:
        if 'fuyu' in model_config.model_name_or_path:
            processor.tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE
        elif 'vicuna' in model_config.model_name_or_path:
            processor.tokenizer.chat_template = VICUNA_CHAT_TEMPLATE
        else:
            raise ValueError("No chat_template found in the tokenizer_config.json, please set the chat_template in the tokenizer_config.json.")
    processor.tokenizer.padding_side = "right"
    
    # Ensure use_cache is set to False
    model.config.use_cache = False


    image_fold = "/home/mc_lmy/datas/10-08_craft-10_dataset/image"
    data_collator = MultimodalDataCollator(processor, image_folder=image_fold,max_seq_length = training_args.max_seq_length)

    ################
    # Dataset
    ################
    
    train_dataset_file = sft_script_args.dataset_name + "-train.json"
    eval_dataset_file = sft_script_args.dataset_name + "-valid.json"

    raw_datasets = load_dataset("json", data_files={"train": train_dataset_file, "validation": eval_dataset_file}, num_proc=8)

    train_dataset = raw_datasets['train']
    eval_dataset = raw_datasets['validation']
    
    ################
    # Optional rich context managers
    ###############
    init_context = nullcontext() if not TRL_USE_RICH else console.status("[bold green]Initializing the SFTTrainer...")
    save_context = (
        nullcontext()
        if not TRL_USE_RICH
        else console.status(f"[bold green]Training completed! Saving the model to {training_args.output_dir}")
    )

    ################
    # Training
    ################
    from rich import print
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        training_args.resume_from_checkpoint = True
        
    with init_context:
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            dataset_text_field="text",  # need a dummy field, UserWarning: You passed a `dataset_text_field` argument to the SFTTrainer, the value you passed will override the one in the `SFTConfig`.
            tokenizer=processor.tokenizer,
            peft_config=get_peft_config(model_config), #if there's no peft config, then return None
            callbacks=[RichProgressCallback] if TRL_USE_RICH else None,
            data_collator=data_collator,
            dataset_kwargs={"skip_prepare_dataset": True}
        )
        
    print_trainable_parameters(trainer.model)

    # trainer.train(resume_from_checkpoint = training_args.resume_from_checkpoint)
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    with save_context:
        trainer.save_model(training_args.output_dir)
        # trainer.push_to_hub()
        # if Accelerator().is_main_process:
        #     processor.push_to_hub(training_args.hub_model_id)
    
