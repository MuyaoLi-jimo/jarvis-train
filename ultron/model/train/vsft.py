import logging
import os
from contextlib import nullcontext
from PIL import Image
import pathlib
import json
import numpy as np
import torch
from torchvision import transforms


TRL_USE_RICH = os.environ.get("TRL_USE_RICH", False)

from trl.commands.cli_utils import init_zero_verbose, SFTScriptArguments, TrlParser

if TRL_USE_RICH:
    init_zero_verbose()
    FORMAT = "%(message)s"
    from rich.console import Console
    from rich.logging import RichHandler


from accelerate import Accelerator
from datasets import load_dataset, Dataset

from tqdm.rich import tqdm
from transformers import AutoTokenizer, AutoProcessor,AutoModelForCausalLM,GenerationConfig
from transformers import FuyuProcessor, LlavaProcessor, Blip2Processor, LlavaNextProcessor
from transformers import LlavaForConditionalGeneration, FuyuForCausalLM, Blip2ForConditionalGeneration, LlavaNextForConditionalGeneration
from torch.optim  import AdamW
from transformers import get_scheduler

from trl import (
    ModelConfig,
    RichProgressCallback,
    SFTConfig,
    SFTTrainer,
    get_peft_config,
    get_quantization_config,
    get_kbit_device_map,
)
from rich import print,console
from ultron.model.train.utils import (
    prepare_conversation_text_with_images,
    prepare_conversation_for_molmo,
    print_trainable_parameters,
    pad_sequence,
    transform_image,
)
from dataclasses import dataclass, field

tqdm.pandas()

if TRL_USE_RICH:
    logging.basicConfig(format=FORMAT, datefmt="[%X]", handlers=[RichHandler()], level=logging.INFO)

################
# Create a data collator to encode text and image pairs
################

class MultimodalDataCollator:
    def __init__(self, processor,model_name_or_path, image_folder = '/nfs-shared/data/JARVIS/tmp/images', with_image = True, resize_image = True, max_seq_length = 1024):
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
        self.no_image_policy = 'random' # 'random' or 'ignore'
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
                else:
                    text = prepare_conversation_text_with_images(example, self.processor.tokenizer)  #合并<image>，并转化为加入chat template的版本
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
                        image=transform_image(image)
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
        check_id = -1 if processor.tokenizer.padding_side=="right" else 0
        if labels[0][check_id].item()!=self.processor.tokenizer.pad_token_id:
            self.my_console.log("[red]Warning! the token length is probably out of max token length!")
        # TODO: add back -- 非常重要
        for label in labels:
            np_label = label.cpu().numpy()
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
        #import pdb; pdb.set_trace()
        return batch

@dataclass
class MoreConfig:
    private_lora_structure:bool = field(
        default=False,
        metadata={"help": "Whether to use adapter or not."},
    )
    

if __name__ == "__main__":
    
    parser = TrlParser((SFTScriptArguments, SFTConfig, ModelConfig,MoreConfig))
    sft_script_args, training_args, model_config,more_config = parser.parse_args_and_config()

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
    VICUNA_CHAT_TEMPLATE = """"{% for message in messages %}{% if message['role'] != 'system' %}{{ message['role'].upper() + ': '}}{% endif %}{# Render all images first #}{% for content in message['content'] | selectattr('type', 'equalto', 'image') %}{{ '<image>\n' }}{% endfor %}{# Render all text next #}{% if message['role'] != 'assistant' %}{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}{{ content['text'] + ' '}}{% endfor %}{% else %}{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}{% generation %}{{ content['text'] + ' '}}{% endgeneration %}{% endfor %}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'ASSISTANT:' }}{% endif %}"}"""
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

    if 'llava-next' in model_config.model_name_or_path or 'llava-v1.6' in model_config.model_name_or_path or 'llava_next' in model_config.model_name_or_path:
        processor_config = dict(
            do_rescale=False,
            patch_size=14,
            vision_feature_select_strategy="default"
        )
        processor = LlavaNextProcessor.from_pretrained(model_config.model_name_or_path,**processor_config)
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
    elif 'molmo' in model_config.model_name_or_path:
        processor_config = dict(
            trust_remote_code=model_config.trust_remote_code,
        )
        processor = AutoProcessor.from_pretrained(model_config.model_name_or_path,**processor_config)
        model = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path, **model_kwargs)  #bf16可以
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

    image_fold = pathlib.Path(sft_script_args.dataset_name).parent
    image_fold = image_fold.parent if image_fold.name=="output" else image_fold
    if 'llava-next' in model_config.model_name_or_path or 'llava_next' in model_config.model_name_or_path or 'llava-v1.6' in model_config.model_name_or_path or "molmo" in model_config.model_name_or_path:
        data_collator = MultimodalDataCollator(processor, image_folder=image_fold,max_seq_length = training_args.max_seq_length,model_name_or_path=model_config.model_name_or_path)
    else:
        raise ValueError(f"be careful! do not write a code for it  {model_config.model_name_or_path}")

    if more_config.private_lora_structure:
        model_config.lora_target_modules = ["embed_tokens","out_proj","k_proj","q_proj","v_proj","o_proj","gate_proj","up_proj","down_proj","lm_head","fc1","fc2","linear_1","linear_2"]

    ################
    # Dataset
    ################
    
    train_dataset_file = sft_script_args.dataset_name + "-train.json"
    eval_dataset_file = sft_script_args.dataset_name + "-valid.json"
    
    raw_datasets = load_dataset("json", data_files={"train": train_dataset_file, "validation": eval_dataset_file}, num_proc=8)

    train_dataset = raw_datasets['train']
    train_dataset = train_dataset.shuffle(27)
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
    # Optimizer
    ################
    # 计算每个epoch的更新步数
    num_train_samples = len(train_dataset)  # 训练集样本总数
    per_device_train_batch_size = training_args.per_device_train_batch_size
    num_train_epochs = training_args.num_train_epochs
    num_devices = training_args.n_gpu if training_args.n_gpu > 0 else 1  # 根据GPU数量调整，无GPU时默认为1

    # 计算每个epoch的更新步数
    from math import ceil
    num_update_steps_per_epoch = ceil(num_train_samples / (per_device_train_batch_size * num_devices))

    # 计算总的训练步数
    total_training_steps = int(num_update_steps_per_epoch * training_args.num_train_epochs)

    # Adjust the total training steps considering max_steps
    if training_args.max_steps > 0:
        total_training_steps = training_args.max_steps

    # 计算预热步数
    if training_args.warmup_steps > 0:
        num_warmup_steps = training_args.warmup_steps
    else:
        num_warmup_steps = int(total_training_steps * training_args.warmup_ratio)

    optimizer = AdamW([
        {
            'params': model.vision_tower.parameters(),
            'lr': training_args.learning_rate * 0.1,
            'weight_decay': training_args.weight_decay,
            'betas': (training_args.adam_beta1, training_args.adam_beta2),
            'eps': training_args.adam_epsilon,
            'correct_bias': False,
        },
        {
            'params': model.multi_modal_projector.parameters(),
            'lr': training_args.learning_rate * 0.1,
            'weight_decay': training_args.weight_decay,
            'betas': (training_args.adam_beta1, training_args.adam_beta2),
            'eps': training_args.adam_epsilon,
            'correct_bias': False,
        },
        {
            'params': model.language_model.parameters(),
            'lr': training_args.learning_rate,
            'weight_decay': training_args.weight_decay,
            'betas': (training_args.adam_beta1, training_args.adam_beta2),
            'eps': training_args.adam_epsilon,
            'correct_bias': False,
        }
    ])  # Note: Set correct_bias=False to follow standard AdamW behavior in Transformers
    
    scheduler = get_scheduler(
        name=training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_training_steps,
        **training_args.lr_scheduler_kwargs,
    )
    ################
    # Training
    ################
    from rich import print
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        training_args.resume_from_checkpoint = True
        

    with init_context:  #使用trl自带的输出增强
        trainer = SFTTrainer(
            model=model,
            optimizers = (optimizer,scheduler),
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            dataset_text_field="text",  # need a dummy field, UserWarning: You passed a `dataset_text_field` argument to the SFTTrainer, the value you passed will override the one in the `SFTConfig`.
            processing_class=processor.tokenizer,
            peft_config=get_peft_config(model_config), #if there's no peft config, then return None
            callbacks=[RichProgressCallback] if TRL_USE_RICH else None,
            data_collator=data_collator,
            dataset_kwargs={"skip_prepare_dataset": True}
        )
    
    print_trainable_parameters(trainer.model,f"model_structure.json")
    #import pdb; pdb.set_trace()

    # trainer.train(resume_from_checkpoint = training_args.resume_from_checkpoint)
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    with save_context:
        
        #trainer.model.save_pretrained(training_args.output_dir/"final_model",save_embedding_layers=True)
        trainer.save_model(training_args.output_dir)
        # trainer.push_to_hub()
        # if Accelerator().is_main_process:
        #     processor.push_to_hub(training_args.hub_model_id)
    
