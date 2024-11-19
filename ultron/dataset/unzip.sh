#!/bin/bash

# 定义目标目录
target_directory="videos"

# 检查目录是否存在，如果不存在则创建
if [ ! -d "$target_directory" ]; then
    mkdir "$target_directory"
fi

# 循环遍历目录中的所有 ZIP 文件
for zip_file in /scratch/mc_lmy/datas/CraftingDataset/contractor/videos1.zip; do
    echo "Processing $zip_file..."
    # 可以在这里添加 unzip -l "$zip_file" 来列出内容
    specific_file="videos/$(basename "$zip_file" .zip).mp4"  # 假设文件名基于 ZIP 文件名

    # 提取特定文件
    if unzip -l "$zip_file" | grep -q "$specific_file"; then
        echo "Extracting $specific_file from $zip_file"
        unzip "$zip_file" "$specific_file" -d "$target_directory"
    else
        echo "$specific_file does not exist in $zip_file"
    fi
done