import json
import pathlib
import os
from typing import Union
import rich
from tqdm import tqdm
import shutil
from datasets import load_dataset
from torch.utils.data import DataLoader
from mc_evaluate.rollout.file_utils import (
    load_json_file,
    dump_json_file,
)

def length_check(args,dataset:list):
    from ultron.model.train.data_collator import MultimodalDataCollator
    
    image_fold = pathlib.Path(args.dataset_path).parent
    image_fold = image_fold.parent if image_fold.name=="output" else image_fold
      
    processor = None
    if 'llava-next' in args.model_path or 'llava-v1.6' in args.model_path or 'llava_next' in args.model_path:
        processor_config = dict(
            do_rescale=False,
            patch_size=14,
            vision_feature_select_strategy="default"
        )
        from transformers import LlavaNextProcessor
        processor = LlavaNextProcessor.from_pretrained(args.model_path,trust_remote_code=True,**processor_config)
    select_length = min(len(dataset),5000)
    select_dataset = dataset.select(range(select_length))
    data_collator = MultimodalDataCollator(processor,model_name_or_path=args.model_path,image_folder=image_fold,check=True) #执行检查
    dataloader = DataLoader(select_dataset, batch_size=64, collate_fn=data_collator, num_workers=8,pin_memory=True,shuffle=True, )
    batch_length_dict = {}
    for d in tqdm(dataloader):
        batch_length_dict.update(d)
    sorted_dict = dict(sorted(batch_length_dict.items(), key=lambda item: item[1], reverse=True))
    dump_json_file(sorted_dict,pathlib.Path(__file__).parent/f"{args.dataset_path.split('/')[-1] + '-len.json'}",if_backup=False)
     
def check_dataset():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', type=str, default='/scratch/mc_lmy/datas/11-25-craft-10-shell_agent-hard/11-25-craft-10-shell_agent-hard-llama-3-h0') 
    parser.add_argument('--model-path', type=str, default='/scratch/mc_lmy/models/llama3-llava-next-8b-hf/') 
    parser.add_argument('--max-seq-length', type=int, default=4096) 
    
    args = parser.parse_args()
    
    dataset_path = args.dataset_path
    
    train_dataset_file = dataset_path + "-train.json"
    eval_dataset_file = dataset_path + "-valid.json"
    
    raw_datasets = load_dataset("json", data_files={"train": train_dataset_file, "validation": eval_dataset_file}, num_proc=8)

    train_dataset = raw_datasets['train']
    eval_dataset = raw_datasets['validation']
    print(f"successfully load {train_dataset_file}")
    length_check(args,train_dataset)
    


if __name__ == "__main__":
    check_dataset()
    #json_file1 = load_json_file("/home/mc_lmy/datas/10-08_craft-10_dataset/embodied_mini_craft_10-10-08-llama-2-train.json")