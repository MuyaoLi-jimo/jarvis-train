'''
Functions:
    - prepare_conversation_text: 
    - print_trainable_parameters:
'''
import logging

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




