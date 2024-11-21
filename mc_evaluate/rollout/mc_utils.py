from typing import List,Tuple
import file_utils

CSS_COLOR = ['blue','green','red','yellow','black','aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige', 'bisque', 'blanchedalmond', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue', 'firebrick', 'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod', 'gray', 'greenyellow', 'grey', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgray', 'lightgreen', 'lightgrey', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightslategrey', 'lightsteelblue', 'lightyellow', 'lime', 'limegreen', 'linen', 'magenta', 'maroon', 'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'navy', 'oldlace', 'olive', 'olivedrab', 'orange', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'purple', 'rebeccapurple', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'silver', 'skyblue', 'slateblue', 'slategray', 'slategrey', 'snow', 'springgreen', 'steelblue', 'tan', 'teal', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'white', 'whitesmoke', 'yellowgreen']

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

def plot_success_rates(datasets: List[Tuple[List[tuple], str, str]], file_path: str):
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    
    plt.figure(figsize=(10, 5))
    
    # Iterate through each dataset
    for idx , (data, label) in enumerate(datasets):
        # Process data: filter by success and sort by steps
        color = CSS_COLOR[idx]
        filtered_data = sorted([step for success, step, _ in data if success])
        
        last_data = 0
        cumulative_percent = []
        new_filtered_data = []
        for i in range(len(filtered_data)-1,-1,-1):
            if last_data==filtered_data[i]:
                continue
            last_data = filtered_data[i]
            new_filtered_data.append(filtered_data[i])
            cumulative_percent.append((i+1) / len(data) * 100)
        cumulative_percent.append(0)
        new_filtered_data.append(0)
        new_filtered_data.reverse()
        cumulative_percent.reverse()
        
        # Plotting
        print(new_filtered_data,cumulative_percent)
        plt.plot(new_filtered_data, cumulative_percent, marker='o', linestyle='-', color=color, label=label)
    
    #plt.xscale('log')
    plt.xlabel('Game Playing Steps (log scale)')
    plt.ylabel('% of Successful Episodes')
    plt.title('Cumulative Success over Game Playing Steps')
    plt.grid(True)
    plt.legend()
    
    plt.savefig(file_path)
    plt.close()  # Close the figure to free up memory


def get_datas(model_name:str,task_name:str):
    from pathlib import Path
    import re
    data_fold = Path(__file__).parent.parent/"temp"
    pattern = re.compile(rf"^{re.escape(model_name)}.*{re.escape(task_name)}$")
    data_paths=[]
    for path in data_fold.iterdir():
        if path.is_dir():
            match = pattern.match(path.name)
            if match:
                label = match.group(0)  # 抽取中间部分作为 label
                data_paths.append((path, label))
    datasets = []
    for data_path,label in data_paths:
        log_path = data_path/"end.json"
        if log_path.exists():
            data = file_utils.load_json_file(log_path)
            datasets.append((data,label))
    return datasets

def draw_whole_pictures():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name',type=str,default='llama3-llava-next-8b-hf-craft-craft_table-shell_agent-hard-llama-3-11-17-1-A100-c4-e3-b16-a4')
    parser.add_argument('--task-name',"-e", type=str, default='jarvis-rt2/craft_crafting_table_multi') #vpt/test_vpt
    args = parser.parse_args()
    datasets = get_datas(args.model_name,args.task_name.split("/")[-1])
    plot_success_rates(datasets,file_path="show.png")

if __name__ == "__main__":
    draw_whole_pictures()
    #show_success_rate([(True, 96, '22'), (True, 110, '4'), (True, 51, '23'), (True, 83, '29'), (True, 130, '10'), (True, 68, '11'), (True, 87, '26'), (True, 162, '8'), (True, 119, '3'), (True, 130, '25'), (True, 124, '0'), (True, 145, '1'), (True, 163, '15'), (True, 192, '5'), (True, 218, '27'), (True, 252, '14'), (True, 213, '13'), (True, 431, '21'), (True, 643, '24'), (False, 1000, '19'), (False, 1000, '20'), (False, 1000, '7'), (False, 1000, '9'), (False, 1000, '28'), (False, 1000, '12'), (False, 1000, '18'), (False, 1000, '17'), (False, 1000, '6'), (False, 1000, '2'), (False, 1000, '16')],
                      #"/scratch/mc_lmy/evaluate/mc-llava_next_llama3_8b-LORA-11-10-craft-craft_table-shell_agent-hard-llama-3-11-13-2-A100-c4-e3-b16-a4-1281_craft_crafting_table_multi/image.png")









