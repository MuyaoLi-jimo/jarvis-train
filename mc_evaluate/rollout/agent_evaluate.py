import argparse
import time
import av
import cv2
import einops
from rich import print,console
from tqdm import tqdm
import jarvis
from jarvis.stark_tech.env_interface import MinecraftWrapper,MyMinecraftWrapper
from mc_evaluate.rollout import agent_wrapper,mc_utils,file_utils

import ray


def resize_image(img, target_resolution = (224, 224)):
    return cv2.resize(img, dsize=target_resolution, interpolation=cv2.INTER_LINEAR)

def evaluate(video_path,checkpoints,environment_config:dict,model_config:dict,device="cuda:0",api_base=None):

    container = av.open(video_path, mode='w', format='mp4')
    stream = container.add_stream('h264', rate=20)
    stream.width = 640
    stream.height = 360
    stream.pix_fmt = 'yuv420p'
    
    if model_config["bpe"]:
        env = MinecraftWrapper(environment_config["env_config"], prev_action_obs=True)
    else:
        env = MyMinecraftWrapper(environment_config["env_config"], prev_action_obs=True)
    agent = None
    if type(api_base)!=type(None):
        agent = agent_wrapper.VLLM_AGENT(checkpoint_path=checkpoints,openai_api_base=api_base,**model_config)
    else:
        agent = agent_wrapper.Agent(checkpoint_path=checkpoints,device=device,**model_config)
        

    instructions = []
    for name, conf in env._env.task_conf.items():
        instructions.append(env._env.task_conf[name]["text"])
        
    obs, info = env.reset()
    from queue import Queue
    fps_queue = Queue()
    success = (False,environment_config["max_frames"])
    for i in range(environment_config["max_frames"]):
        time_start = time.time()
        action = agent.forward([obs["img"]],instructions,verbos=environment_config["verbos"])
        if environment_config["verbos"]:
            console.Console().log(action)
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward>0:
            success = (True,i)
            break
        time_end = time.time()
        curr_fps = 1/(time_end-time_start)
        fps_queue.put(curr_fps)
        if fps_queue.qsize() > 200:
            fps_queue.get()
        average_fps = sum(list(fps_queue.queue))/fps_queue.qsize()
        text = f"frame: {i}, fps: {curr_fps:.2f}, avg_fps: {average_fps:.2f}"
        if i % 50 == 0:
            print(text)
        frame = resize_image(info['pov'], (640, 360))
        action = MyMinecraftWrapper.agent_action_to_env(action)  #转换成env
        for row, (k, v) in enumerate(action.items()):
            
            color = (234, 53, 70) if (v != 0).any() else (249, 200, 14) 
            if k == 'camera':
                v = "[{:.2f}, {:.2f}]".format(v[0], v[1])
            cv2.putText(frame, f"{k}: {v}", (10, 25 + row*15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(frame, text, (150, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (67, 188, 205), 2)
        frame = av.VideoFrame.from_ndarray(frame, format='rgb24')
        for packet in stream.encode(frame):
            container.mux(packet)
    
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    env.close()
    
    return success

@ray.remote
def evaluate_wrapper(video_path,checkpoints,environment_config,api_base,model_config):
    """只使用vllm，因此不需要设备和checkpoints """

    success = evaluate(video_path=video_path,checkpoints=checkpoints,environment_config=environment_config,api_base=api_base,model_config=model_config)
    return (success[0],success[1],video_path.split("/")[-1].split(".")[0])

def multi_evaluate(args):

    ray.init()
    import os
    from pathlib import Path
    
    model_ref_name = args.checkpoints.split('/')[-1]
    if "checkpoint" in model_ref_name:
        checkpoint_num = model_ref_name.split("-")[-1]
        model_base_name = args.checkpoints.split('/')[-2]
        model_ref_name = f"{model_base_name}-{checkpoint_num}"
    
    
    video_fold  = os.path.join(args.video_main_fold, f"{model_ref_name}-{args.env_config.split('/')[-1]}") 
    if not os.path.exists(video_fold):
        Path(video_fold).mkdir(parents=True,exist_ok=True)
    
    model_config = dict(
        temperature=args.temperature,
        history_num = args.history_num,
        instruction_type = args.instruction_type,
        action_chunk_len = args.action_chunk_len,
        bpe = args.bpe,
    )
    environment_config = dict(
        env_config = args.env_config,
        max_frames = args.max_frames,
        verbos = args.verbos,
    )
    
    result_ids = [evaluate_wrapper.remote(video_path=os.path.join(video_fold,f"{i}.mp4"),checkpoints=args.checkpoints,environment_config=environment_config,api_base=args.api_base,model_config=model_config) for i in range(args.workers)]
    futures = result_ids
    resultss = []
    while len(futures) > 0:
        ready_futures, rest_futures = ray.wait(futures,timeout=24*60*60)
        results = ray.get(ready_futures,timeout=60*60)  # Retrieve all results
        resultss.extend(results)
        print(f"part frames IDs: {results} done!")
        futures = rest_futures
    video_log_path = os.path.join(video_fold,"end.json") 
    ray.shutdown()
    
    # 写入日志文件
    file_utils.dump_json_file(resultss,video_log_path)
    mc_utils.show_success_rate(resultss,os.path.join(video_fold,"image.png") )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=6) 
    parser.add_argument('--env-config',"-e", type=str, default='jarvis-rt2/craft_crafting_table') #vpt/test_vpt
    parser.add_argument('--max-frames', type=int, default=200) #vpt/test_vpt
    parser.add_argument('--verbos', type=bool, default=False)
    parser.add_argument('--checkpoints', type=str, default="/scratch/mc_lmy/models/mc_llama3-llava-next-8b-hf-LORA-craft-craft_table-shell_agent-hard-llama-3-h1-11-24-1-A100-c8-e1-b8-a4-200")
    #/home/mc_lmy/model/mc-llava_v1.6_vicuna_mistral_7b-LORA-embodied_mini_craft_10-10-08-llava-v1.6-A100-c4-e3-b16-a4-800") #vpt/test_vpt
    parser.add_argument('--device',type=str,default="cuda:1")
    
    parser.add_argument('--api-base',type=str,default='http://localhost:9206/v1')
    parser.add_argument('--video-main-fold',type=str,default='/scratch/mc_lmy/evaluate')
    
    parser.add_argument('--instruction-type',type=str,default='recipe')
    parser.add_argument('--temperature','-t',type=float,default=0.7)
    parser.add_argument('--history-num',type=int,default=0)
    parser.add_argument('--action-chunk-len',type=int,default=1)
    parser.add_argument('--bpe',type=int,default=0)
    args = parser.parse_args()

    model_config = dict(
        temperature=args.temperature,
        history_num = args.history_num,
        instruction_type = args.instruction_type,
        action_chunk_len = args.action_chunk_len,
        bpe = args.bpe,
    )
    environment_config = dict(
        env_config = args.env_config,
        max_frames = args.max_frames,
        verbos = args.verbos,
    )
    if not args.api_base:
        args.api_base=None
    
    if args.workers==0:
        environment_config["verbos"] = True
        video_path = f"{args.checkpoints.split('/')[-1]}-{args.env_config.split('/')[-1]}.mp4"
        evaluate(video_path=video_path,checkpoints = args.checkpoints,environment_config = environment_config,device=args.device,model_config=model_config)
    elif args.workers==1:
        video_path = f"{args.checkpoints.split('/')[-1]}-{args.env_config.split('/')[-1]}.mp4"
        evaluate(video_path=f"{args.checkpoints.split('/')[-1]}-{args.env_config.split('/')[-1]}.mp4",checkpoints = args.checkpoints,environment_config = environment_config,api_base=args.api_base,model_config=model_config)
    elif args.workers>1:
        multi_evaluate(args)
    
