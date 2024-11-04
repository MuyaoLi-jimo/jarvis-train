'''
Functions:
    - prepare_conversation_text: 
    - print_trainable_parameters:
'''
import logging
from PIL import Image, ImageEnhance, ImageOps
import random


def prepare_conversation_text_with_images(example, tokenizer=None):
    """Prepare the text from a sample of the dataset."""
    conversations = example["conversations"]
    # LLAVA_next: batches处理的时候，输入的是所有images，然后根据<image>来分配
    image_num = len(example["image"]) if isinstance(example["image"],list) else 1
    image_count = 0
    messages = []
    for conv in conversations:
        message_role = conv["role"]
        message_content = ""
        for item in conv["content"]:
            if item["type"] == "text":
                message_content += item["text"]
            elif item["type"] == "image":
                message_content += "<image>"
                image_count+=1
            else:
                print(f"Unknown item type: {item['type']}")
        messages.append({"role": message_role, "content": message_content})
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

def transform_image(image):

    # Randomly adjust hue
    hue_factor = random.uniform(-0.2, 0.2)
    image = ImageEnhance.Color(image).enhance(1 + hue_factor)  # Simulating hue adjustment with Color enhancement

    # Randomly adjust saturation
    saturation_factor = random.uniform(0.8, 1.2)
    image = ImageEnhance.Color(image).enhance(saturation_factor)

    # Randomly adjust brightness
    brightness_factor = random.uniform(0.8, 1.2)
    image = ImageEnhance.Brightness(image).enhance(brightness_factor)

    # Randomly adjust contrast
    contrast_factor = random.uniform(0.8, 1.2)
    image = ImageEnhance.Contrast(image).enhance(contrast_factor)

    # Randomly rotate
    rotate_degree = random.uniform(-2, 2)
    image = image.rotate(rotate_degree, expand=True)

    # Randomly scale
    scale_factor = random.uniform(0.98, 1.02)
    new_size = (int(image.width * scale_factor), int(image.height * scale_factor))
    image = image.resize(new_size, Image.Resampling.LANCZOS)

    # Randomly shear using an affine transformation
    shear_degree = random.uniform(-2, 2)
    a = 1  # x coordinate doesn't change
    b = -shear_degree / 180 * 3.1415927  # y coordinate shift
    c = 0  # x translation
    d = 0  # y coordinate doesn't change
    e = 1
    f = 0  # y translation
    image = image.transform(image.size, Image.AFFINE, (a, b, c, d, e, f))

    # Randomly translate
    max_trans = 2
    trans_x = random.randint(-max_trans, max_trans)
    trans_y = random.randint(-max_trans, max_trans)
    translation_matrix = (1, 0, trans_x, 0, 1, trans_y)
    image = image.transform(image.size, Image.AFFINE, translation_matrix)

    return image

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def chars_token_ratio(dataset, tokenizer, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
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




