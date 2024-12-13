from typing import List,Tuple
from file_utils import load_json_file,dump_json_file

CSS_COLOR = ['blue','green','red','yellow','black', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige', 'bisque', 'blanchedalmond', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue', 'firebrick', 'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod', 'gray', 'greenyellow', 'grey', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgray', 'lightgreen', 'lightgrey', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightslategrey', 'lightsteelblue', 'lightyellow', 'lime', 'limegreen', 'linen', 'magenta', 'maroon', 'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'navy', 'oldlace', 'olive', 'olivedrab', 'orange', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'purple', 'rebeccapurple', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'silver', 'skyblue', 'slateblue', 'slategray', 'slategrey', 'snow', 'springgreen', 'steelblue', 'tan', 'teal', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'white', 'whitesmoke', 'yellowgreen']

from pathlib import Path,PosixPath

def show_success_rate(data:List[tuple],file_path:str):
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    # Process data: filter by success and sort by steps
    filtered_data = sorted([step for success, step, _ in data if success])
    # Create the cumulative percentage list
    last_data = 0
    cumulative_percent = []
    new_filtered_data = []
    for i in range(len(filtered_data)-1,-1,-1):
        if last_data==filtered_data[i]:
            continue
        last_data = filtered_data[i]
        new_filtered_data.append(filtered_data[i])
        cumulative_percent.append((1+i) / len(data) * 100)
    cumulative_percent.append(0)
    new_filtered_data.append(0)
    new_filtered_data.reverse()
    cumulative_percent.reverse()
    # Plotting
    plt.figure(figsize=(10, 5))
    
    plt.plot(new_filtered_data, cumulative_percent, marker='o', linestyle='-', color='b')
    #plt.xscale('log')
    plt.xlabel('Game Playing Steps (log scale)')
    plt.ylabel('% of Successful Episodes')
    plt.title('Cumulative Success over Game Playing Steps')
    plt.grid(True)
    
    plt.savefig(file_path)
    plt.close()  # Close the figure to free up memory

def plot_success_record_inference_steps(model_name:str,task_name:str,success_records: List[Tuple[List[tuple], str]], file_path: str,max_step:int):
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    
    plt.figure(figsize=(10, 5))
    
    # Iterate through each dataset
    for idx , (success_record, inference_step) in enumerate(success_records):
        # Process data: filter by success and sort by steps
        color = CSS_COLOR[idx]
        filtered_data = sorted([step for success, step, _ in success_record if success])
        
        last_data = 0
        cumulative_percent = []
        new_filtered_data = []
        for i in range(len(filtered_data)-1,-1,-1):
            if last_data==filtered_data[i]:
                continue
            last_data = filtered_data[i]
            new_filtered_data.append(filtered_data[i])
            cumulative_percent.append((i+1) / len(success_record) * 100)
        cumulative_percent.append(0)
        new_filtered_data.append(0)
        new_filtered_data.reverse()
        cumulative_percent.reverse()
        new_filtered_data.append(max_step)
        cumulative_percent.append(cumulative_percent[-1])
        
        # Plotting
        plt.plot(new_filtered_data, cumulative_percent, marker='o', linestyle='-', color=color, label=inference_step)
    
    #plt.xscale('log')
    plt.xlabel('Inference Steps')
    plt.ylabel('% of Successful Episodes')
    plt.title(f"Performance of {model_name} \n in {task_name}")
    plt.grid(True)
    plt.legend()
    
    plt.savefig(file_path)
    plt.close()  # Close the figure to free up memory

def plot_success_rates(model_name:str,task_name:str,success_rates:dict,file_path:str):
    import matplotlib.pyplot as plt
    # Data for the new plot

    # Prepare data for plotting
    train_steps = list(success_rates.keys())
    success_rate_values = list(success_rates.values())

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(train_steps, success_rate_values, marker='o', linestyle='-', color='r')

    # Adding title and labels
    plt.title(f"{model_name}'s \nSuccess Rate of Task {task_name} on Train Steps")
    plt.xlabel("Train Steps")
    plt.ylabel("Success Rate")

    # Show grid
    plt.grid(True)

    plt.savefig(file_path)
    plt.close()  # Close the figure to free up memory

def plot_success_rates_on_eval_loss(model_name:str,task_name:str,success_rates:dict,loss_record:dict,file_path:str):
    import matplotlib.pyplot as plt
    train_steps = list(success_rates.keys())
    eval_losses = [loss_record[str(train_step)] for train_step,_ in success_rates.items()]
    success_rate_values = list(success_rates.values())
    
    eval_losses.reverse()
    success_rate_values.reverse()
    
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(eval_losses, success_rate_values, marker='o', linestyle='-', color='b')

    # Adding title and labels
    plt.title(f"{task_name} Success Rate on Eval Loss")
    plt.xlabel("Eval Losses")
    plt.ylabel("Success Rate")

    # Show grid
    plt.grid(True)
    plt.gca().invert_xaxis()
    # Show the plot
    plt.savefig(file_path)
    plt.close()


def plot_eval_loss(model_name:str,loss_record:dict,file_path: str,):
    import matplotlib.pyplot as plt

    # Prepare data for plotting
    steps = list(map(int, loss_record.keys()))
    losses = list(loss_record.values())

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(steps, losses, marker='o', linestyle='-', color='g')

    # Adding title and labels
    plt.title(f"{model_name}'s \nEval Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")

    # Show grid
    plt.grid(True)

    # Show the plot
    plt.savefig(file_path)
    plt.close()


def producing_loss(model_name):
    
    import json
    import ast
    """
    由于离线的wandb存在问题，所以单独处理training loss数据和eval loss，把它们存储成一个List[Dict]格式 
    """
    notes = []
    raw_data_path = Path("mc_evaluate/record/loss_raw")/f"{model_name}.log"
    with open(raw_data_path,"r") as f:
        
        for line in f:
            try:
                safe_line = line.strip()
                if safe_line and safe_line[0]=='{':
                    dict_obj = ast.literal_eval(line.strip())
                    notes.append(dict_obj)
            except:
                continue
    train_loss = {}
    eval_loss= {}    
    current_step = 0  
    current_epoch = -1
    for note in notes:
        if "loss" in note:
            current_step+=1
            if current_epoch > note["epoch"]:
                raise AssertionError(f"check the epoch: {note['epoch']}")
            current_epoch = note["epoch"]
            train_loss[current_step] = note
        elif "eval_loss" in note:
            eval_loss[current_step]=note
    record = {
        "train":train_loss,
        "eval":eval_loss,
    }
    record_data_path = Path("mc_evaluate/record/loss_process")/f"{model_name}.json"
    dump_json_file(record,record_data_path,if_backup=False)

def get_losses(model_name:str):
    from pathlib import Path
    loss_record = load_json_file(Path(__file__).parent.parent/"record"/"loss_process"/f"{model_name}.json")
    eval_losses = loss_record["eval"]
    polished_losses = {}
    for idx,(step,value) in enumerate(eval_losses.items()):
        polished_losses[idx*100] = value
    loss_record["eval"] = polished_losses
    precise_eval_losses = {str(step):value["eval_loss"] for step,value in polished_losses.items()}
    return precise_eval_losses,loss_record

def get_success_record(model_name:str,task_name:str):
    from pathlib import Path
    import re
    data_fold = Path(__file__).parent.parent/"record"/"success_rate"
    pattern = re.compile(rf"^{re.escape(model_name)}(.*?){re.escape(task_name)}$")
    data_paths=[]
    for path in data_fold.iterdir():
        if path.is_dir():
            match = pattern.match(path.name)
            if match:
                label = match.group(1)  # 抽取中间部分作为 label
                if label[0]=='-':
                    label = label[1:]
                if label[-1]=='-':
                    label = label[:-1]                
                data_paths.append((path, int(label)))
    data_paths.sort(key=lambda x: x[1])
    success_records = []
    for data_path,label in data_paths:
        log_path = data_path/"end.json"
        if log_path.exists():
            data = load_json_file(log_path)
            success_records.append((data,label))
    return success_records

def count_success_rate(success_records:List[tuple]):
    success_rates = {}
    for _ , (success_record, inference_step) in enumerate(success_records):
        successes = 0
        for success,_,_ in success_record:
            successes += success
        success_rates[inference_step]= successes/len(success_record)
    return success_rates



def uploading_wandb(model:str,task:str,loss_record:dict,success_rates):
    """上传step-train_loss-eval_loss-success_rate"""
    import wandb
    from tqdm import tqdm
    import os
    import math
    #os.environ["WANDB_API_KEY"] = 
    wandb.login()
    wandb.init(project="VLA_scaling_law-one_stage",name=model)
    
    def key_to_int(src_dict:dict):
        tar_dict = {}
        try:
            for key,value in src_dict.items():
                tar_dict[int(key)] = value
            return tar_dict
        except:
            return src_dict
    
    train_losses = key_to_int(loss_record["train"])
    eval_losses = key_to_int(loss_record["eval"])
    success_rates = key_to_int(success_rates)
    for key,value in tqdm(train_losses.items()):
        if key in eval_losses:
            wandb.log({"eval/"+str(k):v for k,v in eval_losses[key].items()}, commit=False)
        if key in success_rates:
            wandb.log({f"inference/{task}-success_rates":success_rates[key]}, commit=False)
        wandb.log({"train/log-loss":math.log(value["loss"])}, commit=False)
        wandb.log({"train/"+str(k):v for k,v in value.items()})
    wandb.finish()

def draw_whole_pictures(max_step:int):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name',type=str,default='mc_llama3-llava-next-8b-hf-full-11-27-craft-snow_block-shell_agent-hard-llama-3-h0-12-06-1-A100-c4-e3-b16-a4')#mc_llama3-llava-next-8b-hf-full-11-27-craft-crimson_pressure_plate-shell_agent-hard-llama-3-h0-12-04-1-A100-c4-e3-b16-a4')
    parser.add_argument('--task-name',"-e", type=str, default='jarvis-rt2/craft_snow_block_multi') #vpt/test_vpt
    args = parser.parse_args()
    task_name= args.task_name.split("/")[-1]
    # 获取原始的训练数据
    #producing_loss(args.model_name)
    
    # 处理loss数据
    eval_loss_record,train_loss_record = get_losses(args.model_name)
    # 获取原始成功率数据
    success_records = get_success_record(args.model_name,task_name)
    # 处理成功率数据
    success_rates = count_success_rate(success_records)
    uploading_wandb(args.model_name,task_name,train_loss_record,success_rates)
    exit()


    plot_success_rates(args.model_name,task_name,success_rates,file_path="mc_evaluate/record/images/success_rate-with-train_steps.png")
    plot_success_rates_on_eval_loss(args.model_name,task_name,success_rates,eval_loss_record,file_path="mc_evaluate/record/images/success_rate-with-eval_loss.png")
    plot_eval_loss(args.model_name,eval_loss_record,file_path="mc_evaluate/record/images/eval_loss-with-train_steps.png",)
    plot_success_record_inference_steps(args.model_name,task_name,success_records,file_path="mc_evaluate/record/images/success_rate-with-inference_steps.png",max_step=max_step)
        
if __name__ == "__main__":
    draw_whole_pictures(max_step = 600)
    #show_success_rate([(True, 96, '22'), (True, 110, '4'), (True, 51, '23'), (True, 83, '29'), (True, 130, '10'), (True, 68, '11'), (True, 87, '26'), (True, 162, '8'), (True, 119, '3'), (True, 130, '25'), (True, 124, '0'), (True, 145, '1'), (True, 163, '15'), (True, 192, '5'), (True, 218, '27'), (True, 252, '14'), (True, 213, '13'), (True, 431, '21'), (True, 643, '24'), (False, 1000, '19'), (False, 1000, '20'), (False, 1000, '7'), (False, 1000, '9'), (False, 1000, '28'), (False, 1000, '12'), (False, 1000, '18'), (False, 1000, '17'), (False, 1000, '6'), (False, 1000, '2'), (False, 1000, '16')],
                      #"/scratch/mc_lmy/evaluate/mc-llava_next_llama3_8b-LORA-11-10-craft-craft_table-shell_agent-hard-llama-3-11-13-2-A100-c4-e3-b16-a4-1281_craft_crafting_table_multi/image.png")









