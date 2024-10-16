
from ultron.model.inference import action_mapping, input_mapping, load_model

class Agent:
    """只能按照rt2格式运行，早晚要改 """
    def __init__(self,checkpoint_path,device= "cuda:0"):
        self.processor,self.model,self.LLM_backbone,self.VLM_backbone = load_model.load_visual_model(checkpoint_path=checkpoint_path)
        self.processor_wrapper = input_mapping.ProcessorWrapper(self.processor,model_name=self.VLM_backbone)
        self.action_map = action_mapping.ActionMap(tokenizer_type=self.LLM_backbone)
        self.device = device
        
    def forward(self,observations,instruction):
        conversations= []
        conversations.append(self.processor_wrapper.create_message(prompt=instruction))
        text_prompt = self.processor_wrapper.create_text_input(conversations=conversations)
        image = self.processor_wrapper.create_image_input(observations[0])
        inputs = self.processor(images=image, text=text_prompt, return_tensors="pt").to( self.device)
        generate_ids = self.model.to(self.device).generate(**inputs, max_length=1024)
        print(generate_ids)
        outputs = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        print( outputs[outputs.find("[/INST]")+7:])
        #action_token = outputs[outputs.find("[/INST]")+7:]
        
        return #self.action_map.map(generate_ids)
    
if __name__ == "__main__":
    image_path = r"/nfs-shared/pretrain-jarvis-data/retrieve/library/image/003d2195-5780-4a3c-830f-e664843e691d.jpg"
    #from PIL import Image
    #image = Image.open(image_path)
    agent = Agent(checkpoint_path="/nfs-shared-2/limuyao/JARVIS/checkpoints/mc-llava_next_llama3_8b-lora-embodied_mini_craft_table-10-05-A40-c4-e3-b4-a2/checkpoint-1749",device="cuda")
    agent.forward(observations=[image],instruction="create crafting table")