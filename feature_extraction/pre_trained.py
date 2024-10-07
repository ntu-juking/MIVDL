import argparse
from tqdm import tqdm
import json
import numpy as np
import os
from gensim.models import Word2Vec

from transformers import (RobertaConfig, RobertaModel, RobertaTokenizer)

MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)}

def merge_tokens(file_path, output_file):
    datas = json.load(open(file_path))
    output_data = []
    for ix, data in enumerate(tqdm(datas)):
        graph = data['graph']
        original_tokens = data['original_tokens']
        # symbolic_tokens = data['symbolic_tokens']
        # targets = data['targets']
        path = []
        # symbolic_path = []
        # flows = []
        # symbolic_flows = [] b

        for indices in graph:
            for idx in indices:
                if idx < len(original_tokens):
                    path.append(original_tokens[idx])
                    # for flow in original_tokens[idx]:
                    #     flows.append(flow)
            # for idx in indices:
            #     if idx < len(symbolic_tokens):
            #         symbolic_path.append(symbolic_tokens[idx])
                    # for flow in symbolic_tokens[idx]:
                    #     symbolic_flows.append(flow)
        output = {
            # 'target': targets,
            'path': path,
            # 'symbolic_path': symbolic_path,
            # 'flow': flows,
            # 'symbolic_flow': symbolic_flows
        }
        output_data.append(output)
    # with open(output_file, 'w') as of:
    #     json.dump(output_data, of)
    #     of.close()
    print('processing finished!!!!!!!!!!!!')
    return output_data

def flatten_and_truncate(path_values):
    for path, values in path_values.items():
        new_values = []
        for sublist in values:
            # 将二维列表展开成一维列表
            flattened_list = [str(item) for item in sublist]  # 将列表中的元素转换为字符串
            flattened_string = ' '.join(flattened_list)  # 将列表转换为字符串
            # 截取字符串的前 length 个字符
            # 将处理后的值添加到新列表
            new_values.append(flattened_string)
        # 更新原集合中的值为处理后的结果
        merged_string = ' '.join(new_values)
    return merged_string

def tokenize(datas,tokenizer, path, max_length = 512, padding = False):
    print("tokenizing--------------------------")
    tokens_ids = []
    # prompt_text = "This code and graph is [MASK]."
    prompt_code = "This is code."
    prompt_path = "This is path."
    prompt_code = tokenizer.tokenize(prompt_code)
    prompt_path = tokenizer.tokenize(prompt_path)
    for idx, data in enumerate(tqdm(datas)):
        tokens_code = tokenizer.tokenize(data['code'])
        tokens_code = tokens_code[:max_length - 4]
        token_len = len(tokens_code)
        truncated_length = max_length - 4 - token_len
        tokens = [tokenizer.cls_token] + prompt_code +[tokenizer.sep_token]
        tokens = tokens + tokens_code + [tokenizer.sep_token]
        tokens = tokens + prompt_path + [tokenizer.sep_token]
        tokens_path = path[idx]
        merged_string = flatten_and_truncate(tokens_path)
        merged_string = tokenizer.tokenize(merged_string)
        tokens = tokens + merged_string[:truncated_length]
        tokens_id = tokenizer.convert_tokens_to_ids(tokens)
        print("tokens_id:", tokens_id)
        tokens_ids.append(tokens_id)
    return tokens_ids

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', default='Devign')
    parser.add_argument('--output', default='../processed_data/pre_trained_feature/')
    parser.add_argument('--output_file', default='../processed_data/', type=str, help="The merge file.")
    parser.add_argument('--model_type', default='roberta', type=str, help="The model architecture to be fine-tuned.")
    parser.add_argument('--model_name', default='../microsoft/graphcodebert-base', type=str, help="The model architecture to be fine-tuned.")
    args = parser.parse_args()

    json_file_path = '../processed_data/' + args.project + '_full_data_with_slices.json'
    slice_file_path = '../processed_data/' + args.project + '-line-ggnn.json'
    output_file = args.output_file + args.project + '-pretrained.json'
    path = merge_tokens(slice_file_path, output_file)
    data = json.load(open(json_file_path))
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.model_name)
    model = model_class.from_pretrained(args.model_name)
    config = config_class.from_pretrained(args.model_name)
    max_length = config.max_position_embeddings
    tokens_ids = tokenize(data, path, tokenizer, max_length, True)
