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
    image_num = len(example["image"]) if isinstance(example["image"],list) else 1
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

def print_trainable_parameters(model,record_path = None):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    model_shapes = []
    for name, param in model.named_parameters():
        model_shapes.append([param.requires_grad,name,param.shape])
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    import json
    if record_path:
        with open(record_path,mode="w",encoding="UTF-8") as f:
            json.dump(model_shapes, f, indent=4)
        
        with open(record_path.replace(".json","-scratch.txt"),mode="w",encoding="UTF-8") as f:
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




