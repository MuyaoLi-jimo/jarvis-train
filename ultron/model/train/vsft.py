import logging
import os
from contextlib import nullcontext
import pathlib


TRL_USE_RICH = os.environ.get("TRL_USE_RICH", False)

from trl.commands.cli_utils import init_zero_verbose, SFTScriptArguments, TrlParser

if TRL_USE_RICH:
    init_zero_verbose()
    FORMAT = "%(message)s"
    from rich.console import Console
    from rich.logging import RichHandler

import inspect
from datasets import load_dataset

import torch
from tqdm.rich import tqdm
from transformers import AutoTokenizer, AutoProcessor,AutoModelForCausalLM,GenerationConfig
from transformers import FuyuProcessor, LlavaProcessor, Blip2Processor, LlavaNextProcessor
from transformers import LlavaForConditionalGeneration, FuyuForCausalLM, Blip2ForConditionalGeneration, LlavaNextForConditionalGeneration
from transformers import Trainer
from peft import prepare_model_for_kbit_training,get_peft_model
from trl import (
    ModelConfig,
    RichProgressCallback,
    SFTConfig,
    SFTTrainer,
    get_peft_config,
    get_quantization_config,
    get_kbit_device_map,
)
from trl.trainer.utils import peft_module_casting_to_bf16
from omegaconf import OmegaConf
from rich import print,console
from ultron.model.train.utils import (
    print_trainable_parameters,
    prepare_optimizer_scheduler,
)
from ultron.model.train.data_collator import MultimodalDataCollator
import dataclasses
from dataclasses import dataclass, field

tqdm.pandas()

if TRL_USE_RICH:
    logging.basicConfig(format=FORMAT, datefmt="[%X]", handlers=[RichHandler()], level=logging.INFO)

@dataclass
class MoreConfig:
    cfg_file:str = field(
        default="config/base.yaml",
        metadata={"help": "the config file path of specific setting"},
    )
    

if __name__ == "__main__":
    
    parser = TrlParser((SFTScriptArguments, SFTConfig, ModelConfig,MoreConfig))
    sft_script_args, training_args, model_config,more_config = parser.parse_args_and_config()
    
    file_name = pathlib.Path(__file__).parent
    special_cfg = OmegaConf.load(file_name/more_config.cfg_file)
    if more_config.cfg_file != "config/base.yaml":
        base_config = OmegaConf.load(file_name/"config/base.yaml")
        special_cfg = OmegaConf.merge(base_config, special_cfg)

    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    # Force use our print callback
    if TRL_USE_RICH:
        training_args.disable_tqdm = True
        console = Console()

    ################
    # Model, Tokenizer & Processor
    ################
    ### discard: if no chat_template is defined in tokenizer_config.json, use the default one
    DEFAULT_CHAT_TEMPLATE = """{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = message['role'] + ':\n\n'+ message['content'] + '\n' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}"""
    VICUNA_CHAT_TEMPLATE = """{{ bos_token }}{% for message in messages %}{% if message['role'] != 'system' %}{{ ' '+message['role'].upper() + ': '}}{% endif %}{# Render all images first #}{% for content in message['content'] | selectattr('type', 'equalto', 'image') %}{{ '<image>\n' }}{% endfor %}{# Render all text next #}{% if message['role'] != 'assistant' %}{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}{{ content['text'] + '\n'}}{% endfor %}{% else %}{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}{% generation %}{{ content['text'] + eos_token + '\n' }}{% endgeneration %}{% endfor %}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'ASSISTANT:' }}{% endif %}"""
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    if 'llava-next' in model_config.model_name_or_path or 'llava-v1.6' in model_config.model_name_or_path or 'llava_next' in model_config.model_name_or_path:
        processor_config = dict(
            do_rescale=False,
            patch_size=14,
            vision_feature_select_strategy="default"
        )
        processor = LlavaNextProcessor.from_pretrained(model_config.model_name_or_path,**processor_config)
        model = LlavaNextForConditionalGeneration.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    elif 'llava-1.5' in model_config.model_name_or_path or 'llava-gemma' in model_config.model_name_or_path:
        processor = LlavaProcessor.from_pretrained(model_config.model_name_or_path)
        model = LlavaForConditionalGeneration.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    elif 'fuyu' in model_config.model_name_or_path:
        processor = FuyuProcessor.from_pretrained(model_config.model_name_or_path)
        model = FuyuForCausalLM.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    elif 'blip2' in model_config.model_name_or_path:
        processor = Blip2Processor.from_pretrained(model_config.model_name_or_path)
        model = Blip2ForConditionalGeneration.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    elif 'molmo' in model_config.model_name_or_path:
        processor_config = dict(
            trust_remote_code=model_config.trust_remote_code,
        )
        processor = AutoProcessor.from_pretrained(model_config.model_name_or_path,**processor_config)
        model = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path, **model_kwargs)  #bf16可以
    else:
        processor = AutoProcessor.from_pretrained(model_config.model_name_or_path)
    
    if not processor.tokenizer.chat_template:
        if 'fuyu' in model_config.model_name_or_path:
            processor.tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE
        else:
            raise ValueError("No chat_template found in the tokenizer_config.json, please set the chat_template in the tokenizer_config.json.")
    if 'vicuna' in model_config.model_name_or_path:
        processor.tokenizer.chat_template = VICUNA_CHAT_TEMPLATE
    
    processor.tokenizer.padding_side = "right"
    if getattr(processor.tokenizer, "pad_token", None) is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    #############
    #  PEFT
    #############
    if model_config.use_peft:
        model_config.lora_target_modules = list(special_cfg.model.lora_target_modules)
        peft_config=get_peft_config(model_config)
        
        _support_gc_kwargs = hasattr( training_args, "gradient_checkpointing_kwargs") and "gradient_checkpointing_kwargs" in list(inspect.signature(prepare_model_for_kbit_training).parameters)
        gradient_checkpointing_kwargs = getattr(training_args, "gradient_checkpointing_kwargs", None) or {}
        is_sharded_qlora = False
        # Below is to support QLoRA + FSDP / DS-Zero3 - one should never call
        # peft_module_casting_to_bf16 or prepare_model_for_kbit_training when doing
        # QLoRA + FSDP / DS-Zero3
        if getattr(model, "is_loaded_in_4bit", False):
            for _, param in model.named_parameters():
                if param.__class__.__name__ == "Params4bit":
                    is_sharded_qlora = param.data.device.type in {"cpu", "meta"}
                    break
        if getattr(model, "is_loaded_in_8bit", False) or (getattr(model, "is_loaded_in_4bit", False) and not is_sharded_qlora):
            prepare_model_kwargs = {
                "use_gradient_checkpointing": getattr(training_args, "gradient_checkpointing", False)
            }

            if _support_gc_kwargs:
                prepare_model_kwargs["gradient_checkpointing_kwargs"] = gradient_checkpointing_kwargs

            model = prepare_model_for_kbit_training(model, **prepare_model_kwargs)

            if training_args is not None:
                training_args = dataclasses.replace(training_args, gradient_checkpointing=False)
        elif getattr(training_args, "gradient_checkpointing", False) and (
            "use_reentrant" not in gradient_checkpointing_kwargs
            or gradient_checkpointing_kwargs["use_reentrant"]
        ):
            # For backward compatibility with older versions of transformers
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        if (
            "autocast_adapter_dtype" in list(inspect.signature(get_peft_model).parameters)
            and getattr(model, "is_loaded_in_4bit", False)
            and is_sharded_qlora
        ):
            model = get_peft_model(model, peft_config, autocast_adapter_dtype=False)
        else:
            model = get_peft_model(model, peft_config)
        if (training_args.bf16 and getattr(model, "is_loaded_in_4bit", False) and not is_sharded_qlora):
            peft_module_casting_to_bf16(model)
        

    
    ##################
    # DataCollator
    ##################

    # 找到image_fold
    image_fold = pathlib.Path(sft_script_args.dataset_name).parent
    image_fold = image_fold.parent if image_fold.name=="output" else image_fold
    if 'llava-next' in model_config.model_name_or_path or 'llava_next' in model_config.model_name_or_path or 'llava-v1.6' in model_config.model_name_or_path or "molmo" in model_config.model_name_or_path:
        data_collator = MultimodalDataCollator(processor, image_folder=image_fold,max_seq_length = training_args.max_seq_length,model_name_or_path=model_config.model_name_or_path)
    else:
        raise ValueError(f"be careful! do not write a code for it  {model_config.model_name_or_path}")


    ################
    # Dataset
    ################
    
    train_dataset_file = sft_script_args.dataset_name + "-train.json"
    eval_dataset_file = sft_script_args.dataset_name + "-valid.json"
    
    raw_datasets = load_dataset("json", data_files={"train": train_dataset_file, "validation": eval_dataset_file}, num_proc=8)

    train_dataset = raw_datasets['train']
    train_dataset = train_dataset.shuffle(27)
    eval_dataset = raw_datasets['validation']
    
    ################
    # Optional rich context managers
    ###############
    init_context = nullcontext() if not TRL_USE_RICH else console.status("[bold green]Initializing the SFTTrainer...")
    save_context = (
        nullcontext()
        if not TRL_USE_RICH
        else console.status(f"[bold green]Training completed! Saving the model to {training_args.output_dir}")
    )

    ################
    # Training
    ################
    from rich import print
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        training_args.resume_from_checkpoint = True
        
    # Ensure use_cache is set to False
    model.config.use_cache = False   
    
    if special_cfg.train.use_sfttrainer:  
             
        with init_context:  #使用trl自带的输出增强
            trainer = SFTTrainer(
                model=model,
                #optimizers = prepare_optimizer_scheduler(model,len(train_dataset),special_cfg,training_args),
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                dataset_text_field="text",  # need a dummy field, UserWarning: You passed a `dataset_text_field` argument to the SFTTrainer, the value you passed will override the one in the `SFTConfig`.
                processing_class=processor.tokenizer,
                peft_config=get_peft_config(model_config), #if there's no peft config, then return None
                callbacks=[RichProgressCallback] if TRL_USE_RICH else None,
                data_collator=data_collator,
                dataset_kwargs={"skip_prepare_dataset": True}
            )

    else: 
        training_args.dataset_text_field = "text"
        training_args.dataset_kwargs = {"skip_prepare_dataset": True}

        trainer = Trainer( #MyTrainer(
            model=model,
            args=training_args,#special_cfg=special_cfg, 
            #optimizers=prepare_optimizer_scheduler(model,len(train_dataset),special_cfg,training_args),
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processor.tokenizer,
            model_init=None,
            compute_metrics=None,
            callbacks=[RichProgressCallback] if TRL_USE_RICH else None,
            preprocess_logits_for_metrics=None,
        )

    print_trainable_parameters(trainer.model,trainer.optimizer,f"model_structure.json")
    #import pdb; pdb.set_trace()

    # trainer.train(resume_from_checkpoint = training_args.resume_from_checkpoint)
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    with save_context:
        
        #trainer.model.save_pretrained(training_args.output_dir/"final_model",save_embedding_layers=True)
        trainer.save_model(training_args.output_dir)
        # trainer.push_to_hub()
        # if Accelerator().is_main_process:
        #     processor.push_to_hub(training_args.hub_model_id)
    
