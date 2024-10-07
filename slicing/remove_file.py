import os
import shutil

def move_files(source_dir,source_dir_1):
    # print("source_dir_1:", source_dir_1)
    # print("source_dir:", source_dir)
    # Get a list of files in the source directory
    source_dir = os.path.join('./Fan_csv', source_dir)
    source_dir_1 = os.path.join('./Fan_csv', source_dir_1)
    files = os.listdir(source_dir)

    # The target directory is the higher-level directory
    destination_dir = os.path.abspath(os.path.join(source_dir, '../', '..'))
    for file_name in files:
        # The complete path of the source file
        source_file = os.path.join(source_dir, file_name)

        # If it is a file instead of a directory, move to the target directory
        if os.path.isfile(source_file):
            shutil.move(source_file, destination_dir)
            # print(f"Moved {file_name} to {destination_dir}")
    # Delete tmp directory
    tmp_dir = os.path.join(source_dir_1, 'tmp')
    # print("tmp_dir:",tmp_dir)
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
        # print(f"Removed {tmp_dir}")

# Source directory path
current_directory = os.getcwd()
directory = './Fan_csv'
folders = [folder for folder in os.listdir(directory) if os.path.isdir(os.path.join(directory, folder))]
print(folders[:2])
for folder in folders:
    source_directory = folder + "/tmp/" + folder
    source_directory_1 = folder + "/"
    # move file
    move_files(source_directory,source_directory_1)