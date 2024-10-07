import os

def delete_pdf_files(folder_path):
    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a folder。")
        return

    for item in os.listdir(folder_path):
        if item.endswith(".pdf"):
            file_path = os.path.join(folder_path, item)
            try:
                os.remove(file_path)  # 删除文件
                print(f"deleted: {file_path}")
            except OSError as e:
                print(f"Error: delete error: {file_path} - {e}")

# 指定要删除pdf文件的文件夹路径
folder_to_clean = "../code-slicer/slice-output/"

# 调用函数删除pdf文件
delete_pdf_files(folder_to_clean)
