#!/bin/bash
# chmod +x delete_global_step.sh
# ./delete_global_step.sh /path/to/your/directory
# 检查是否提供了文件夹路径
if [ -z "$1" ]; then
  echo "Usage: $0 <directory>"
  exit 1
fi

# 获取输入的文件夹路径
DIR=$1

# 遍历文件夹中的所有以 checkpoint- 开头的文件夹
for checkpoint_dir in "$DIR"/checkpoint-*; do
  # 检查是否为目录
  if [ -d "$checkpoint_dir" ]; then
    
    # 提取 checkpoint 后的编号
    number=$(basename "$checkpoint_dir" | grep -oP '(?<=checkpoint-)\d+')
    # 删除 global_step$number 文件
    target_file="$checkpoint_dir/global_step$number"
    rm -rf $target_file
    echo $target_file
  fi
done