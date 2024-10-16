import argparse
import time
import av
import cv2
from rich import print
from jarvis.stark_tech.env_interface import MinecraftWrapper
import evaluate.rollout.agent_wrapper as agent_wrapper


def resize_image(img, target_resolution = (224, 224)):
    return cv2.resize(img, dsize=target_resolution, interpolation=cv2.INTER_LINEAR)


def evaluate(args):
    container = av.open("env_test.mp4", mode='w', format='mp4')
    stream = container.add_stream('h264', rate=20)
    stream.width = 640
    stream.height = 360
    stream.pix_fmt = 'yuv420p'
    
    env = MinecraftWrapper(args.env, prev_action_obs=True)
    #agent = agent_wrapper.Agent(checkpoint_path=args.checkpoints,device=args.device)
    
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
    for i in range(600):
        time_start = time.time()
        #action = agent.forward()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        # import ipdb; ipdb.set_trace()
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='jarvis-rt2/base') #vpt/test_vpt
    parser.add_argument('--checkpoints', type=str, default='') #vpt/test_vpt
    parser.add_argument('--device',type=str,default="cuda:0")
    args = parser.parse_args()
    evaluate(args)