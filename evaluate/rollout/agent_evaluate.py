import argparse
import time
import av
import cv2
import einops
from rich import print
from tqdm import tqdm
from jarvis.stark_tech.env_interface import MinecraftWrapper
import evaluate.rollout.agent_wrapper as agent_wrapper
import ray

def resize_image(img, target_resolution = (224, 224)):
    return cv2.resize(img, dsize=target_resolution, interpolation=cv2.INTER_LINEAR)


def evaluate(video_path,checkpoints,env_config,model_config:dict,device="cuda:0",api_base=None,verbos=False):

    container = av.open(video_path, mode='w', format='mp4')
    stream = container.add_stream('h264', rate=20)
    stream.width = 640
    stream.height = 360
    stream.pix_fmt = 'yuv420p'
    
    env = MinecraftWrapper(env_config, prev_action_obs=True)
    agent = None
    if type(api_base)!=type(None):
        agent = agent_wrapper.VLLM_AGENT(checkpoint_path=checkpoints,openai_api_base=api_base,**model_config)
    else:
        agent = agent_wrapper.Agent(checkpoint_path=checkpoints,device=device,**model_config)
        

    instructions = []
    for name, conf in env._env.task_conf.items():
        instructions.append(env._env.task_conf[name]["text"])
        
    obs, info = env.reset()
    """ 
    OrderedDict([('buttons', array([2040])), ('camera', array([56]))])
    {
        'img': array([], dtype=uint8),
        'prev_action': {
            'attack': array(0),
            'back': array(0),
            'forward': array(1),
            'jump': array(0),
            'left': array(0),
            'right': array(0),
            'sneak': array(0),
            'sprint': array(0),
            'use': array(1),
            'drop': array(1),
            'inventory': array(0),
            'hotbar.1': array(0),
            'hotbar.2': array(1),
            'hotbar.3': array(0),
            'hotbar.4': array(0),
            'hotbar.5': array(0),
            'hotbar.6': array(0),
            'hotbar.7': array(0),
            'hotbar.8': array(0),
            'hotbar.9': array(0),
            'camera': array([0., 0.])
        },
        'text': 'craft item crafting table',
        'obs_conf': None
    }
    """
    from queue import Queue
    fps_queue = Queue()
    for i in range(200):
        time_start = time.time()
        action = agent.forward([obs["img"]],instructions,verbos=verbos)
        if verbos:
            print(action)
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated:
            obs, info = env.reset()
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
        action = MinecraftWrapper.agent_action_to_env(action)  #转换成env
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

@ray.remote
def evaluate_wrapper(video_path,checkpoints,env_config,api_base,verbos,model_config):
    """只使用vllm，因此不需要设备和checkpoints """

    evaluate(video_path=video_path,checkpoints=checkpoints,env_config=env_config,api_base=api_base,verbos=verbos,model_config=model_config)
    print(video_path.split("/")[-1].split(".")[0])
    return video_path.split("/")[-1].split(".")[0]

def multi_evaluate(args):

    ray.init()
    import os
    from pathlib import Path
    video_fold  = os.path.join(args.video_main_fold, args.checkpoints.split("/")[-1])
    if not os.path.exists(video_fold):
        Path(video_fold).mkdir(parents=True,exist_ok=True)
    
    model_config = dict(
        temperature=args.temperature
    )

    result_ids = [evaluate_wrapper.remote(video_path=os.path.join(video_fold,f"{i}.mp4"),checkpoints=args.checkpoints,env_config=args.env_config,api_base=args.api_base,verbos=False,model_config=model_config) for i in range(args.workers)]
    futures = result_ids
    while len(futures) > 0:
        ready_futures, rest_futures = ray.wait(futures,timeout=24*60*60)
        results = ray.get(ready_futures,timeout=60*60)  # Retrieve all results
        print(f"part frames IDs: {results} done!")
        futures = rest_futures
    ray.shutdown()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=6) 
    parser.add_argument('--env_config', type=str, default='jarvis-rt2/craft_crafting_table') #vpt/test_vpt
    parser.add_argument('--checkpoints', type=str, default="/scratch/mc_lmy/models/mc-llava_v1.6_mistral_7b-LORA-embodied_mini_10-30-craft-craft_table-shell_agent-normal-mistral-10-30-A100-c4-e10-b16-a1-576")
    #/home/mc_lmy/model/mc-llava_v1.6_vicuna_mistral_7b-LORA-embodied_mini_craft_10-10-08-llava-v1.6-A100-c4-e3-b16-a4-800") #vpt/test_vpt
    parser.add_argument('--device',type=str,default="cuda:7")
    parser.add_argument('--api_base',type=str,default='http://localhost:9003/v1')
    parser.add_argument('--video_main_fold',type=str,default='/scratch/mc_lmy/evaluate')

    parser.add_argument('--temperature',type=int,default=0.5)
    args = parser.parse_args()

    model_config = dict(
        temperature=args.temperature
    )

    if not args.api_base:
        args.api_base=None
    if args.workers==0:
        evaluate(video_path=f"{args.checkpoints.split('/')[-1]}.mp4",checkpoints = args.checkpoints,env_config = args.env_config,device=args.device,verbos=True,model_config=model_config)
    elif args.workers==1:
        evaluate(video_path=f"{args.checkpoints.split('/')[-1]}.mp4",checkpoints = args.checkpoints,env_config = args.env_config,api_base=args.api_base,model_config=model_config)
    elif args.workers>1:
        multi_evaluate(args)
    
