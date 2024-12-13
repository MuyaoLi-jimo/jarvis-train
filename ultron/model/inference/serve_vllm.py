import argparse
from pathlib import Path,PosixPath
from typing import Union
import os
import rich
import re
import time
import subprocess
import signal

def load_txt_file(file_path:Union[str , PosixPath]):
    
    if isinstance(file_path,PosixPath):
        file_path = str(file_path)
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                txt_file = f.read()
        except IOError as e:
            rich.print(f"[red]无法打开文件：{e}[/red]")
    else:
         rich.print(f"[yellow]{file_path}文件不存在，传入空文件[/yellow]")

    return txt_file

def dump_txt_file(file,file_path:Union[str , PosixPath],if_print = True):
    if isinstance(file_path,PosixPath):
        file_path = str(file_path)
    before_exist = os.path.exists(file_path)
    try:
        with open(file_path, 'w') as f:
            # 使用pickle.dump将数据序列化到文件
            f.write(str(file))
        if before_exist and if_print:
            rich.print(f"[yellow]更新{file_path}[/yellow]")
        elif if_print:
            rich.print(f"[green]创建{file_path}[/green]")
    except IOError as e:
        rich.print(f"[red]文件写入失败：{e}[/red]")

def get_gpu_usage():
    """check usage off gpu"""
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'], 
                            stdout=subprocess.PIPE, 
                            encoding='utf-8')
    gpu_info = result.stdout.strip().split('\n')
    
    gpu_usages = []
    for idx, info in enumerate(gpu_info):
        used, total = map(int, info.split(','))
        gpu_usages.append((idx, used, total))
    
    return gpu_usages

def get_avaliable_gpus(cuda_num:int):
    avaliable_gpus = []
    gpu_usages = get_gpu_usage()
    for gpu_usage in gpu_usages:
        idx, used, total = gpu_usage
        if used/total < 0.03:
            avaliable_gpus.append(str(idx))
            if len(avaliable_gpus)>=cuda_num:
                break
    if len(avaliable_gpus)<cuda_num:
        print("there aren't enough avaliable GPUs to use")
        print(f"[red]{gpu_usages}")
        raise Exception
    return avaliable_gpus

def run_vllm_server(devices:list,device_num:int,model_path, 
                    log_path,port, 
                    max_model_len, 
                    gpu_memory_utilization,
                    limit_mm_per_prompt:int=1,
                    chat_template:str=""):
    if devices==[]:
        devices = get_avaliable_gpus(device_num)
    devices_str = ','.join(devices)
    device_num = len(devices_str.split(','))
    if not log_path:
        log_path = Path(__file__).parent/"logs"/f"{Path(model_path).name}.log"
    dump_txt_file("",log_path)
    
    # 构建命令
    
    command = f"CUDA_VISIBLE_DEVICES={devices_str} nohup vllm serve {model_path} --port {port} --max-model-len {max_model_len} --gpu-memory-utilization {gpu_memory_utilization} --trust-remote-code"
    if chat_template!="":
        command += f" --chat-template {chat_template}"
    if device_num>1:
        command += f" --tensor-parallel-size {device_num}"
    if limit_mm_per_prompt>1:
        command += f" --limit-mm-per-prompt image={limit_mm_per_prompt}"
    command += f" > {log_path} 2>&1 &"
    print(command)
    # 使用 shell=True 来运行 nohup 命令
    subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # 获取进程的 PID,使用这个方法是为了保证vllm加载的模型已经开始运行
    while True:
        log_text = load_txt_file(log_path)
        pid = extract_pid(log_text)
        if type(pid) != type(None):
            break
        print("[yellow]not found server yet")
        time.sleep(10)
    
    return int(pid),devices_str

def stop_vllm_server(pid:int=0,log_path=""):
    if pid == 0:
        log_text = load_txt_file(log_path)
        pid = extract_pid(log_text)
        if not pid:
            return
    os.kill(pid, signal.SIGINT)

def extract_pid(text):
    # Regex pattern to match the line with the PID
    pattern = r"Started server process \[(\d+)\]"
    match = re.search(pattern, text)
    if match:
        return int(match.group(1))  
    return None 


def str_to_list(value):
    """
    Converts a comma-separated string to a list of integers.
    Example: "0,1" -> [0, 1]
    """
    return [str(x) for x in value.split(",")]

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--card-num", type=int, default=2)
    parser.add_argument(
        "--cuda-visible-devices", 
        type=str_to_list, 
        default=[0, 1], 
        help="Comma-separated list of CUDA devices to use, e.g., '0,1,2'. Default is [0,1]."
    )
    parser.add_argument("--limit-mm-per-prompt", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--port", type=int, default=5209)
    parser.add_argument("--model-path", type=str, default="/scratch/mc_lmy/models/mc-llava_next_llama3_8b-lora-11-10-craft-craft_table-shell_agent-hard-llama-3-11-16-1-A100-c4-e3-b16-a1-3600")
    parser.add_argument("--chat-template", type=str, default="")
    parser.add_argument("--start", type=bool, default=False,help="True then start, kill then end")

    args = parser.parse_args()
    log_path = Path(__file__).parent/"log"/f"{args.model_path.split('/')[-1]}.log"
    log_path.parent.mkdir(parents=True,exist_ok=True)
    model_path = args.model_path
    limit_mm_per_prompt = args.limit_mm_per_prompt
    chat_template = ""
    if "vicuna" in model_path:
        limit_mm_per_prompt = 1
    if args.chat_template:
        chat_template = args.chat_template
    if args.start:
        run_vllm_server(devices = args.cuda_visible_devices,
                        device_num=args.card_num,
                        model_path=model_path,
                        port=args.port,
                        max_model_len=args.max_model_len,
                        log_path=log_path,
                        gpu_memory_utilization=0.95,
                        limit_mm_per_prompt=limit_mm_per_prompt,
                        chat_template = chat_template,)
    else:
        stop_vllm_server(log_path=log_path)
        
        