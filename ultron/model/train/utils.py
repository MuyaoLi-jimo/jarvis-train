'''
Functions:
    - prepare_conversation_text: 
    - print_trainable_parameters:
'''
import logging
from PIL import Image, ImageEnhance, ImageOps
import random
from rich import console
from typing import List


def prepare_conversation_text_with_images(example, tokenizer=None,simple_messages=True):
    """Prepare the text from a sample of the dataset."""
    conversations = example["conversations"]
    # LLAVA_next: batches处理的时候，输入的是所有images，然后根据<image>来分配
    images = example.get("image",[])
    image_num = len(images) if isinstance(images,list) else 1
    image_count = 0
    messages = []
    for conv in conversations:
        message_role = conv["role"]
        message_content = ""
        if simple_messages:
            for item in conv["content"]:
                if item["type"] == "text":
                    message_content += item["text"]
                elif item["type"] == "image":
                    message_content += "<image>"
                    image_count+=1
                else:
                    print(f"Unknown item type: {item['type']}")
            messages.append({"role": message_role, "content": message_content})
        else:
            for item in conv["content"]:
                if item["type"] == "image":
                    message_content += "<image>"
                    image_count+=1
            messages.append(conv)
    assert image_num == image_count
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return text


def prepare_conversation_for_molmo(example,tokenizer=None):
    conversations = example["conversations"]
    roles = ["user","assistant"]
    assert len(conversations)==2
    text = []
    image_count = 0
    for idx,conv in enumerate(conversations):
        assert conv["role"]==roles[idx]
        for item in conv["content"]:
            if item["type"] == "text":
                text.append(item["text"])
            elif item["type"] == "image":
                image_count+=1
    text.append(image_count)
    return text
        
def pad_sequence(sequences, padding_value,max_length=4096,padding_side='right',truncation=True):
    """
    Pad a list of sequences to the same length.
    sequences: list of tensors in [seq_len, *] shape
    """
    assert padding_side in ['right', 'left'], "padding_side must be either 'right' or 'left'"
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max_length if truncation else max(max_length,max(len(seq) for seq in sequences))
    batch_size = len(sequences)
    output = sequences[0].new_full((batch_size, max_len) + trailing_dims, padding_value)
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        if truncation and length > max_len:
            seq = seq[:max_len]
            length = max_len
        if padding_side == 'right':
            output[i, :length] = seq
        else:
            output[i, -length:] = seq
    return output
    

from PIL import Image, ImageEnhance, ImageOps
import random

def hue_augmentation(image:Image.Image,random_hue:float=0.05)->Image.Image:
    # Randomly adjust hue
    hue_factor = random.uniform(-random_hue, random_hue)
    image = ImageEnhance.Color(image).enhance(1 + hue_factor)  # Simulating hue adjustment with Color enhancement
    return image

def saturation_augmentation(image: Image.Image, random_saturation: List[float] = [0.8, 1.2]) -> Image.Image:
    # Randomly adjust saturation within a given range
    saturation_factor = random.uniform(*random_saturation)
    image = ImageEnhance.Color(image).enhance(saturation_factor)
    return image

def brightness_augmentation(image: Image.Image, random_brightness: List[float] = [0.8, 1.2]) -> Image.Image:
    # Randomly adjust brightness within a given range
    brightness_factor = random.uniform(*random_brightness)
    image = ImageEnhance.Brightness(image).enhance(brightness_factor)
    return image

def contrast_augmentation(image: Image.Image, random_contrast: List[float] = [0.8, 1.2]) -> Image.Image:
    # Randomly adjust contrast within a given range
    contrast_factor = random.uniform(*random_contrast)
    image = ImageEnhance.Contrast(image).enhance(contrast_factor)
    return image

def rotate_augmentation(image: Image.Image, random_rotate: List[float] = [-2, 2]) -> Image.Image:
    # Randomly rotate within a given degree range
    rotate_degree = random.uniform(*random_rotate)
    image = image.rotate(rotate_degree, expand=True)
    return image

def scale_augmentation(image: Image.Image, scale_range: List[float] = [0.98, 1.02]) -> Image.Image:
    # Randomly scale the image by a factor within the given range
    scale_factor = random.uniform(*scale_range)
    new_size = (int(image.width * scale_factor), int(image.height * scale_factor))
    image = image.resize(new_size, Image.Resampling.LANCZOS)
    return image

def random_shear(image: Image.Image, shear_range: float = 2) -> Image.Image:
    # Randomly shear the image using an affine transformation
    shear_degree = random.uniform(-shear_range, shear_range)
    radians = shear_degree / 180 * 3.1415927  # Convert degrees to radians
    a = 1  # x coordinate doesn't change
    b = -shear_degree / 180 * 3.1415927  # y coordinate shift
    c = 0  # x translation
    d = 0  # y coordinate doesn't change
    e = 1
    f = 0  # y translation
    return image.transform(image.size, Image.AFFINE, (a, b, c, d, e, f))

def random_translate(image: Image.Image, max_trans: int = 2) -> Image.Image:
    # Randomly translate the image
    trans_x = random.randint(-max_trans, max_trans)
    trans_y = random.randint(-max_trans, max_trans)
    translation_matrix = (1, 0, trans_x, 0, 1, trans_y)
    return image.transform(image.size, Image.AFFINE, translation_matrix)



def transform_image(image:Image.Image)->Image.Image:
    #TODO: 把image增强的参数引入
    # Randomly adjust hue
    image = hue_augmentation(image) 

    # Randomly adjust saturation
    image = saturation_augmentation(image)

    # Randomly adjust brightness
    image = brightness_augmentation(image)

    # Randomly adjust contrast
    image = contrast_augmentation(image)

    # Randomly rotate
    #image = rotate_augmentation(image)

    # Randomly scale
    #image = scale_augmentation(image)

    # Randomly shear using an affine transformation
    #image = random_shear(image)

    # Randomly translate
    image = random_translate(image)

    return image

import re
import torch
from typing import List
from omegaconf import OmegaConf
from trl import SFTConfig
def get_parameters_by_regex(
    model: torch.nn.Module, 
    regex: str, 
    decay_parameters: List[str] = [], 
    decay: bool = True
):
    """
    根据正则表达式筛选模型中的参数。

    Args:
        model: 模型对象
        regex: 正则表达式字符串，用于匹配参数名称。
        decay_parameters: 包含需要进行 decay 的参数名称列表。
        decay: 是否启用 decay 的匹配条件。

    Returns:
        List[torch.nn.Parameter]: 匹配的参数列表。
    """
    return [
        parameter for name, parameter in model.named_parameters()
        if parameter.requires_grad and re.search(regex, name) and 
           ((decay and name in decay_parameters) or (not decay and name not in decay_parameters))
    ]

def get_optimizer_param_groups_settings(
    model: torch.nn.Module,
    special_cfg: OmegaConf,
    training_args: SFTConfig,
    decay_parameters: List[str] = []
):
    """ 
    通过 config 匹配查询，将模型不同组件分类为 optimizer_param_groups，准备传递给 Optimizer。
    
    Args:
        model: 模型对象。
        special_cfg: 包含 optimizer 参数配置的配置文件。
        training_args: 包含训练参数的配置对象。
        decay_parameters: 包含需要进行 decay 的参数名称列表。
    
    Returns:
        List[dict]: 包含 optimizer 参数组设置的列表。
    """
    optimizer_param_groups_settings = []
    for optimizer_param_group in special_cfg.train.optimizer_param_groups:
        # 匹配需要 decay 的参数
        decay_params = get_parameters_by_regex(
            model=model,
            regex=optimizer_param_group.params_regex,
            decay_parameters=decay_parameters,
            decay=True
        )
        optimizer_param_groups_settings.append({
            'params': decay_params,
            'lr': training_args.learning_rate * optimizer_param_group.lr_multiplier,
            "weight_decay": training_args.weight_decay,
        })

        # 匹配不需要 decay 的参数
        non_decay_params = get_parameters_by_regex(
            model=model,
            regex=optimizer_param_group.params_regex,
            decay_parameters=decay_parameters,
            decay=False
        )
        optimizer_param_groups_settings.append({
            'params': non_decay_params,
            'lr': training_args.learning_rate * optimizer_param_group.lr_multiplier,
            "weight_decay": 0.0,
        })
    return optimizer_param_groups_settings

from torch.optim  import AdamW
from transformers import get_scheduler
def prepare_optimizer_scheduler(
        model:torch.nn.Module,
        num_train_samples:int,
        special_cfg:OmegaConf,
        training_args:SFTConfig,
    ):

    optimizer = AdamW(get_optimizer_param_groups_settings(model,special_cfg,training_args),  
        weight_decay =training_args.weight_decay,
        betas= (training_args.adam_beta1, training_args.adam_beta2),
        eps= training_args.adam_epsilon,)  # Note: Set correct_bias=False to follow standard AdamW behavior in Transformers
    
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
    
    scheduler = get_scheduler(
        name=training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_training_steps,
        **training_args.lr_scheduler_kwargs,
    )
    
    return optimizer,scheduler

def print_trainable_parameters(model:torch.nn.Module,optimizer:torch.optim.Optimizer=None,record_path = None):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    model_shapes = []
    for name, parameter in model.named_parameters():
        if optimizer:
            optimizer_group_idx = None
            for idx,param_group in enumerate(optimizer.param_groups):
                for param in param_group["params"]:
                    if parameter is param:
                        optimizer_group_idx = idx
            model_shapes.append([parameter.requires_grad,name,parameter.shape,optimizer_group_idx])
        else:
            model_shapes.append([parameter.requires_grad,name,parameter.shape])
        all_param += parameter.numel()
        if parameter.requires_grad:
            trainable_params += parameter.numel()
    import json
    if record_path:
        with open(record_path,mode="w",encoding="UTF-8") as f:
            json.dump(model_shapes, f, indent=4)
        
        with open(record_path.replace(".json","-scratch.txt"),mode="w",encoding="UTF-8") as f:
            print(optimizer, file=f)
            print(model, file=f)
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def chars_token_ratio(dataset, tokenizer, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    from tqdm import tqdm
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        # text = prepare_sample_text(example)
        if 'text' in example.keys():
            text = example['text']
        elif 'conversations' in example.keys():
            text = prepare_conversation_text_with_images(example, tokenizer)
        else:
            print('No text found in example')
            continue
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))

    return total_characters / total_tokens


from transformers import Trainer
from transformers.utils import is_sagemaker_mp_enabled,logging
import torch.nn as nn
import smp

class MyTrainer(Trainer):
    def __init__(self,special_cfg:OmegaConf, *args, **kwargs):
    # 调用父类 Trainer 的初始化方法
        super().__init__(*args, **kwargs)
        self.special_cfg = special_cfg
    
    def create_optimizer(self):
            """
            Setup the optimizer.

            We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
            Trainer's init through `optimizers`, or subclass and override this method in a subclass.
            """
            opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

            if self.optimizer is None:
                decay_parameters = self.get_decay_parameter_names(opt_model)
                optimizer_grouped_parameters = get_optimizer_param_groups_settings(opt_model,self.special_cfg,self.args,decay_parameters)
                
                optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)

                # Overwrite `params` in case it's created by `get_optimizer_cls_and_kwargs`
                # e.g. for GaLore optimizer.
                if "params" in optimizer_kwargs:
                    optimizer_grouped_parameters = optimizer_kwargs.pop("params")

                # Overwrite `model` in case it's created by `get_optimizer_cls_and_kwargs`
                # e.g. for LOMO optimizer.
                if "model" in optimizer_kwargs:
                    optimizer_grouped_parameters = optimizer_kwargs.pop("model")

                # For layer-wise dummy optimizers we overwrite optimizer_grouped_parameters with `optimizer_dict`
                # to avoid arguments conflicts.
                if "optimizer_dict" in optimizer_kwargs:
                    optimizer_grouped_parameters = optimizer_kwargs.pop("optimizer_dict")
                    
                
                self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

                if optimizer_cls.__name__ == "Adam8bit":
                    import bitsandbytes

                    manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                    skipped = 0
                    for module in opt_model.modules():
                        if isinstance(module, nn.Embedding):
                            skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                            
                            manager.register_module_override(module, "weight", {"optim_bits": 32})

            if is_sagemaker_mp_enabled():
                self.optimizer = smp.DistributedOptimizer(self.optimizer)
                
            return self.optimizer


