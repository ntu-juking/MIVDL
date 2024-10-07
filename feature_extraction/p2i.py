import os
from pdf2image import convert_from_path
from PIL import Image, ImageFile

# 增加图像处理库的限制
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None  # 移除默认的图像大小限制

folder = "../code-slicer/slice-output/"

i = 0
for filename in os.listdir(folder):
    if filename.endswith(".pdf"):
        path = os.path.join(folder,filename)
        pages = convert_from_path(path, 300)  # 300 is the resolution in DPI
        i += 1
        new_filename = filename[:-6]
        for z, page in enumerate(pages):
            page.save(f'../code-slicer/png/{new_filename}.png', 'PNG')
        print(f'第{i}个png')