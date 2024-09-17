import os
import shutil

def move_files(source_dir,source_dir_1):
    # print("source_dir_1:", source_dir_1)
    # print("source_dir:", source_dir)
    # 获取源目录下的文件列表
    files = os.listdir(source_dir)
    # print(files)
    # 目标目录为上上一级目录
    destination_dir = os.path.abspath(os.path.join(source_dir, '../data/', '..'))
    for file_name in files:
        # 源文件的完整路径
        source_file = os.path.join(source_dir, file_name)

        # 如果是文件而不是目录，则移动到目标目录
        if os.path.isfile(source_file):
            shutil.move(source_file, destination_dir)
            print(f"Moved {file_name} to {destination_dir}")
    # 删除tmp目录
    tmp_dir = os.path.join(source_dir_1, 'tmp')
    # print("tmp_dir:",tmp_dir)
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
        print(f"Removed {tmp_dir}")

# 源目录路径
current_directory = os.getcwd()
folders = [folder for folder in os.listdir(current_directory) if os.path.isdir(os.path.join(current_directory, folder))]
for folder in folders:
    source_directory = folder + "/tmp/" + folder
    source_directory_1 = folder + "/"
    # 移动文件
    move_files(source_directory,source_directory_1)
    # print(source_directory)
    # print(source_directory_1)