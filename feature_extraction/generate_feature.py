import torch
import pandas as pd
import json
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from openprompt.plms import get_model_class

# 设置设备
device = torch.device("cuda")

model_class = get_model_class(plm_type='roberta')
model_path = "../microsoft/graphcodebert-base/"
# model_path = "./models/Devign/"
config = model_class.config.from_pretrained(model_path)
tokenizer = model_class.tokenizer.from_pretrained(model_path)
model = model_class.model.from_pretrained(model_path)

# 移动模型到设备
model = model.to(device)

def flatten_and_truncate(path_values):
    for path, values in path_values.items():
        new_values = []
        for sublist in values:
            # 将二维列表展开成一维列表
            flattened_list = [str(item) for item in sublist]  # 将列表中的元素转换为字符串
            flattened_string = ' '.join(flattened_list)  # 将列表转换为字符串
            # 将处理后的值添加到新列表
            new_values.append(flattened_string)
        # 更新原集合中的值为处理后的结果
        merged_string = ' '.join(new_values)
        path_values[path] = [merged_string]
    return path_values

def load_datasets(path):
    from tqdm import tqdm
    print("reading text--------------------------")
    # 读取 Excel 文件
    global df
    for idx, data in enumerate(tqdm(path)):
        merged_string = flatten_and_truncate(data)
        # print("merged_string:",merged_string)
        df_tmp = pd.DataFrame(merged_string)
        df = df.append(df_tmp)
    print("shape:", df.shape)
# 定义数据预处理函数

def merge_tokens(file_path):
    datas = json.load(open(file_path))
    output_data = []
    for ix, data in enumerate(tqdm(datas)):
        graph = data['graph']
        original_tokens = data['original_tokens']
        target = data['targets']
        path = []
        for indices in graph:
            for idx in indices:
                if idx < len(original_tokens):
                    path.append(original_tokens[idx])
        output = {
            'path': path,
            'code': original_tokens,
            'target': target
        }
        output_data.append(output)
    print('processing finished!!!!!!!!!!!!')
    return output_data

def preprocess_data(code, path, labels):
    input_ids = []
    attention_masks = []
    texts = code + path
    for text in texts:
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids.append(encoding['input_ids'])
        attention_masks.append(encoding['attention_mask'])
    labels = labels.to_list()
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)
    return input_ids, attention_masks, labels

path = '../processed_data/Devign-line-ggnn.json'
path_1 = merge_tokens(path)
load_datasets(path_1)

# 预处理数据
input_ids, attention_masks, labels = preprocess_data(df['code'], df['path'], df['target'])

# 创建数据加载器
train_data = torch.utils.data.TensorDataset(input_ids, attention_masks, labels)
dataloader = DataLoader(train_data, batch_size=16, shuffle=True)

#计算训练步数
total_steps = len(dataloader)
progress_bar = tqdm(range(total_steps))
for batch in dataloader:
    batch = tuple(t.to(device) for t in batch)
    input_ids, attention_masks, labels = batch
    outputs = model(input_ids, attention_mask=attention_masks)
    print(outputs.shape)