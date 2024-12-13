import time
from ultron.model.inference import action_mapping, load_model, processor_wrapper
from rich import print
from openai import OpenAI
import random
from typing import Literal
import copy

class Agent:
    """只能按照rt2格式运行，早晚要改 """
    def __init__(self,checkpoint_path,history_num=0,device= "cuda:0",temperature=0.5):
        self.model_path = checkpoint_path
        self.processor,self.model,self.LLM_backbone,self.VLM_backbone = load_model.load_visual_model(device=device,checkpoint_path=checkpoint_path)
        self.processor_wrapper = processor_wrapper.ProcessorWrapper(self.processor,model_name=self.VLM_backbone)
        self.history_num = history_num
        self.history = []
        self.action_map = action_mapping.OneActionTokenizer(tokenizer_type=self.LLM_backbone)
        self.device = device
        self.temperature = temperature
        
    def forward(self,observations,instructions,verbos=False):
        """
        输入image,输出action len(instructions)==1
        """
        conversations= []
        for instruction in instructions: 
            conversations.append(self.processor_wrapper.create_message(prompt=instruction))
        text_prompt = self.processor_wrapper.create_text_input(conversations=conversations)

        if "mistral" in  self.model_path:
            text_prompt += "ಮ"
        elif "vicuna_mistral" in  self.model_path:
            text_prompt += "ಮ"
        elif "vicuna" in self.model_path:
            text_prompt += "유"

        image = self.processor_wrapper.create_image_input(observations[0])
        inputs = self.processor(images=image, text=text_prompt, return_tensors="pt").to( self.device)
        generate_ids = self.model.to(self.device).generate(**inputs, max_new_tokens=64,temperature=self.temperature,do_sample=True)
        
        if "vicuna_mistral" in  self.model_path:
            outputs = self.processor.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
            actions =  self.action_map.decode(outputs)
        else:
            if verbos:
                outputs = self.processor.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
                print(outputs)
                print(generate_ids)
            actions = self.action_map.decode(generate_ids)
        return actions
    
    def analyze_forward(self,observations,instructions):
        conversations= []
        for instruction in instructions: 
            conversations.append(self.processor_wrapper.create_message(prompt=instruction))
        start_time = time.time()
        # 创建文本输入
        start = time.time()
        text_prompt = self.processor_wrapper.create_text_input(conversations=conversations)
        end = time.time()
        print("Creating text input took {:.3f} seconds.".format(end - start))

        # 特定模型路径条件的处理
        if "vicuna_mistral" in self.model_path:
            text_prompt += "ಮ"

        # 创建图像输入
        start = time.time()
        image = self.processor_wrapper.create_image_input(observations[0])
        end = time.time()
        print("Creating image input took {:.3f} seconds.".format(end - start))

        # 创建模型输入并移至设备
        start = time.time()
        inputs = self.processor(images=image, text=text_prompt, return_tensors="pt").to(self.device)
        end = time.time()
        print("Preparing and moving inputs took {:.3f} seconds.".format(end - start))

        # 模型生成操作
        start = time.time()
        generate_ids = self.model.to(self.device).generate(**inputs, max_new_tokens=128)
        end = time.time()
        print("Model generating took {:.3f} seconds.".format(end - start))

        # 输出处理和动作映射
        if "vicuna_mistral" in self.model_path:
            start = time.time()
            outputs = self.processor.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
            print(outputs)
            actions = self.action_map.decode(outputs)
            end = time.time()
            print("Processing outputs and mapping actions for 'vicuna_mistral' took {:.3f} seconds.".format(end - start))
        else:
            start = time.time()
            actions = self.action_map.decode(generate_ids)
            end = time.time()
            print("Mapping actions took {:.3f} seconds.".format(end - start))

        # 记录并输出整段代码的总执行时间
        total_time = time.time() - start_time
        print("Total execution time: {:.3f} seconds.".format(total_time))

        return actions

class VLLM_AGENT:
    def __init__(self,checkpoint_path,openai_api_base,openai_api_key="EMPTY",
                 history_num=0,action_chunk_len=1,bpe=0,
                 device= "cuda:0",
                instruction_type:Literal['simple','recipe'] = 'recipe',
                temperature=0.5):
        
        self.LLM_backbone,self.VLM_backbone = load_model.load_visual_model(device=device,checkpoint_path=checkpoint_path,quick_load=True)
        self.bpe = bpe
        if bpe:
            self.action_tokenizer = action_mapping.BPEActionTokenizer(tokenizer_type=self.LLM_backbone,)
        else:
            self.action_tokenizer = action_mapping.OneActionTokenizer(tokenizer_type=self.LLM_backbone)
        from file_utils import load_json_file
        from pathlib import Path
        self.text_library = load_json_file(Path(__file__).parent/"assert"/"instructions.json")
        self.processor_wrapper = processor_wrapper.ProcessorWrapper(None,model_name=self.VLM_backbone)
        # 一次返回一个action chunk
        self.action_chunk_len=action_chunk_len
        self.actions = []
        # 用于带有记忆的agent
        self.history_num = history_num
        
        self.history = []
        self.client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )
        models = self.client.models.list()
        self.model = models.data[0].id
        self.temperature = temperature
        self.instruction_type = instruction_type
        self.tokenizer = None
        if self.LLM_backbone=="llama-3" or self.LLM_backbone=="llama-2":
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                checkpoint_path,  
                trust_remote_code=True,
            )
    def reset(self):
        self.history = []
        
    def create_instruction(self,text):
        return random.choice(self.text_library[text].get("instruct"))
        
    def create_thought(self,text):
        return copy.copy(self.text_library[text].get("thought",text))
        
    def add_prompt(self,text):
        prompt =None
        if self.instruction_type == 'recipe':
            prompt = self.create_instruction(text)
            recipe = self.text_library[text].get("recipe")
            if recipe:
                prompt += "\nArrange the materials in the crafting grid according to the following pattern: \n"
                prompt += recipe[0]
                # 为了保证之前的模型依然能用
                for jdx in range(1,len(recipe)):
                    prompt +="\nEach " + recipe[jdx][0] + " represents a " + recipe[jdx][1] + ".\n"
        elif self.instruction_type == 'simple':
            prompt = self.create_thought(text) #因为老版和新版不太一样
        else:
            raise ValueError(f"do not set the instruction class {self.instruction_type}")
        return prompt
        
    def forward(self,observations,instructions,verbos=False):
        if self.actions:
            if verbos:
                print(self.actions)
            return self.actions.pop(0)
        messages = []
        image = self.processor_wrapper.create_image_input(observations[0]) 
        prompts = [self.add_prompt(instructions[0])]
        thought= self.create_thought(instructions[0])
        images = [image]
        if self.history_num:
            if not self.history: #如果历史为空
                self.history = [(image,self.action_tokenizer.null_token()[0],copy.copy(thought),0)]*self.history_num
            new_history = [None]*self.history_num
            new_history[:-1] = self.history[1:]
            history_images = []
            for im,action,past_thought,_ in self.history:
                prompts[-1] = prompts[-1]+"\nthought: " + past_thought + "\nobservation: "  #往上一个prompt上加上这一步的thought
                history_images.append(im)
                prompts.append("\naction: "+ action)
            history_images.append(image)
            images = history_images
        if self.instruction_type != 'simple':
            prompts[-1]  = prompts[-1]+"\nthought: " + thought + "\nobservation: "
        if verbos:
            print(prompts)
        messages.append(self.processor_wrapper.create_message_vllm(prompt=prompts,image=images))
        open_logprobs = True if verbos else False
        chat_completion = self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            temperature=self.temperature,
            max_tokens=1024,
            logprobs = open_logprobs,
        )
        # ! should use tokenizer to transform it back to token_ids
        if verbos:
            print(chat_completion)
        outputs = chat_completion.choices[0].message.content
        if self.history_num:
            new_history[-1] = (image,outputs,thought,self.history[-1][-1]+1)
            self.history = new_history
        if self.LLM_backbone=="llama-3" or self.LLM_backbone=="llama-2":
            # outputs = "<|reserved_special_token_178|><|reserved_special_token_231|><|reserved_special_token_209|><|reserved_special_token_179|>"
            outputs = self.tokenizer(outputs)["input_ids"]
        if verbos:
            if self.history_num:
                print("idx of this token: ",self.history[-1][-1])
            print(outputs)

        actions =  self.action_tokenizer.decode(outputs)

        if not self.bpe:
            len_action = min(self.action_chunk_len,len(actions))
            self.actions = actions[:len_action]
        else:
            self.actions = actions
        if verbos:
            print(self.actions)
        return self.actions.pop(0)

if __name__ == "__main__":
    from PIL import Image
    image_path = r"/home/limuyao/datas/jarvis-dataset-003/image/602a3447-ecce-47f1-8964-31b39ae27399888.jpg"
    image = Image.open(image_path)
    openai_api_base='http://localhost:9204/v1'
    checkpoint_path= "/scratch/mc_lmy/models/llama3-llava-next-8b-hf"
    ##
    agent = VLLM_AGENT(checkpoint_path=checkpoint_path,openai_api_base=openai_api_base)
    #print(agent.forward(observations=[image],instructions=["crafting a crafting table"]))
    #exit()
    
    #agent = Agent(checkpoint_path=checkpoint_path,device="cuda:4")
    print(agent.forward(observations=[image],instructions=["points at a crafting table"],verbos=True))