
    
            
        


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name',type=str,default='mc_llama3-llava-next-8b-hf-full-craft-craft_table-shell_agent-hard-llama-3-11-22-1-A100-c4-e3-b16-a4')#mc_llama3-llava-next-8b-hf-full-11-27-craft-crimson_pressure_plate-shell_agent-hard-llama-3-h0-12-04-1-A100-c4-e3-b16-a4')
    parser.add_argument('--task-name',"-e", type=str, default='jarvis-rt2/craft_snow_block_multi') #vpt/test_vpt
    args = parser.parse_args()

    producing_loss(args.model_name)