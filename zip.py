import os
import zipfile

def zipdir(path, ziph):
    # ziph 是 zipfile 句柄
    for root, dirs, files in os.walk(path):
        for file in files:
            # 创建一个基于根目录的相对路径
            ziph.write(os.path.join(root, file),
                       os.path.relpath(os.path.join(root, file),
                                       os.path.join(path, '..')))

# 打开一个 zip 文件以写入
with zipfile.ZipFile('/home/mc_lmy/mark2.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
    # 添加不同的目录到 ZIP 文件
    #zipdir('/home/mc_lmy/workspace/Mark2', zipf)
    #zipdir('/home/mc_lmy/workspace/MC_VLA_dataset', zipf)
    zipdir('/home/mc_lmy/miniconda3/envs/mark2', zipf)