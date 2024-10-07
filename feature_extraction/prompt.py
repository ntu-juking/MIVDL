import os
import torch
import json
from tqdm import tqdm
import pandas as pd
from datasets import load_dataset, Dataset
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate,ManualVerbalizer
from openprompt import PromptForClassification
from openprompt.data_utils import InputExample
import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
seed = 40
batch_size = 16
num_class = 2
max_seq_l = 512
lr = 2e-5
num_epochs = 8
use_cuda = True
model_name = "roberta"
pretrainedmodel_path = "./pretrained-roberta-large"  # the path of the pre-trained model

def merge_tokens(file_path):
    datas = json.load(open(file_path))
    output_data = []
    for ix, data in enumerate(tqdm(datas)):
        graph = data['graph']
        original_tokens = data['original_tokens']
        path = []

        for indices in graph:
            for idx in indices:
                if idx < len(original_tokens):
                    path.append(original_tokens[idx])
        output = {
            'path': path,
        }
        output_data.append(output)
    print('processing finished!!!!!!!!!!!!')
    return output_data

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
        path_values[path] = merged_string
    return path_values

def tokenize(datas,tokenizer, path, max_length = 512, padding = False):
    print("tokenizing--------------------------")
    tokens_ids = []
    # prompt_text = "This code and graph is [MASK]."
    prompt_code = ['This','is code.']
    prompt_path = ['This', 'is', 'path.']
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

def Classification(datas, pre_file, path):
    from tqdm import tqdm
    print("reading text--------------------------")
    # 读取 Excel 文件
    df = pd.DataFrame(columns=['code', 'target','path'])
    for idx, data in enumerate(tqdm(datas)):
        tokens_path = path[idx]
        merged_string = flatten_and_truncate(tokens_path)
        merged_string = pd.Series([merged_string])
        df.append(data['code'], data['target'], tokens_path[0])

    # t_dataset = df.loc[df['target'] == 1]
    # f_dataset = df.loc[df['target'] == 0]
    # datasets = f_dataset.append(t_dataset)
    # datasets = datasets[["modified_column", "target"]]
    dataset = Dataset.from_dict(df)

    traintest = dataset.train_test_split(test_size=0.2, seed=seed)
    # traintest_20 = traintest['train'].train_test_split(test_size=0.75,seed=seed)
    validationtest = traintest['test'].train_test_split(test_size=0.5, seed=seed)
    train_val_test = {}
    train_val_test['train'] = traintest['train']
    train_val_test['validation'] = validationtest['train']
    train_val_test['test'] = validationtest['test']

    dataset = {}
    for split in ['train', 'validation', 'test']:
        dataset[split] = []
        for data in train_val_test[split]:
            input_example = InputExample(text_a=data['code'], text_b =data['path'],label=int(data['target']))
            dataset[split].append(input_example)

    # load plm
    plm, tokenizer, model_config, WrapperClass = load_plm("roberta", pre_file)

    # construct hard template
    # template_text = 'The vulnerability is {"mask"}. {"placeholder":"text_a"}'
    # template_text = 'Here is {"mask"} vulnerability. {"placeholder":"text_a"}'
    # template_text = '{"placeholder":"text_a"} The severity is {"mask"}.'
    template_text = 'This code {"placeholder":"text_a"} and path {"placeholder":"text_b"} is a vulnerability. {"mask"}.'
    mytemplate = ManualTemplate(tokenizer=tokenizer, text=template_text)

    # define the verbalizer
    myverbalizer = ManualVerbalizer(tokenizer, num_classes=num_class, label_words=[["true", "yes"], ["false", "no"]])
    # myverbalizer = ManualVerbalizer(tokenizer, num_classes=num_class, label_words=[["low"], ["medium"], ["high"], ["critical"]])

    # define prompt model for classification
    prompt_model = PromptForClassification(plm=plm, template=mytemplate, verbalizer=myverbalizer, freeze_plm=False)
    if use_cuda:
        prompt_model = prompt_model.cuda()

    # DataLoader
    from openprompt import PromptDataLoader
    train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer,
                                        tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l,
                                        batch_size=batch_size, shuffle=True,
                                        teacher_forcing=False, predict_eos_token=False, truncate_method="head")
    validation_dataloader = PromptDataLoader(dataset=dataset["validation"], template=mytemplate, tokenizer=tokenizer,
                                             tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l,
                                             batch_size=batch_size, shuffle=True,
                                             teacher_forcing=False, predict_eos_token=False, truncate_method="head")
    test_dataloader = PromptDataLoader(dataset=dataset["test"], template=mytemplate, tokenizer=tokenizer,
                                       tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l,
                                       batch_size=batch_size, shuffle=True,
                                       teacher_forcing=False, predict_eos_token=False, truncate_method="head")

    from transformers import AdamW, get_linear_schedule_with_warmup

    loss_func = torch.nn.CrossEntropyLoss()
    no_decay = ['bias', 'LayerNorm.weight']
    # it's always good practice to set no decay to biase and LayerNorm parameters
    optimizer_grouped_parameters = [
        {'params': [p for n, p in prompt_model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in prompt_model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    num_training_steps = num_epochs * len(train_dataloader)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    def test(prompt_model, test_dataloader):
        allpreds = []
        alllabels = []
        with torch.no_grad():
            for step, inputs in enumerate(test_dataloader):
                if use_cuda:
                    inputs = inputs.cuda()
                logits = prompt_model(inputs)
                labels = inputs['label']
                alllabels.extend(labels.cpu().tolist())
                allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
            acc = accuracy_score(alllabels, allpreds)
            f1 = f1_score(alllabels, allpreds)
            precision = precision_score(alllabels, allpreds, zero_division=0)
            recall = recall_score(alllabels, allpreds)
            print("acc: {}  recall: {}  precision: {}  f1: {}".format(acc, recall, precision, f1))
        return acc, recall, precision, f1

    from tqdm.auto import tqdm

    output_dir = "../hard_prompt"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    progress_bar = tqdm(range(num_training_steps))

    bestRecall = 0
    bestAcc = 0
    bestPre = 0
    bestF1 = 0
    result_Recall = 0
    result_F1 = 0
    result_Pre = 0
    result_Acc = 0
    for epoch in range(num_epochs):
        # train
        tot_loss = 0
        for step, inputs in enumerate(train_dataloader):
            if use_cuda:
                inputs = inputs.cuda()
            logits = prompt_model(inputs)
            labels = inputs['label']
            loss = loss_func(logits, labels)
            loss.backward()
            tot_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            progress_bar.update(1)
        print("\nEpoch {}, average loss: {}".format(epoch, tot_loss / (step + 1)), flush=True)

        # validate
        print('\n\nepoch{}------------validate------------'.format(epoch))
        acc, recall, precision, f1 = test(prompt_model, validation_dataloader)
        if recall > bestRecall:
            bestRecall = recall
        if acc > bestAcc:
            bestAcc = acc
        if precision > bestPre:
            bestPre = precision
        if f1 > bestF1:
            bestF1 = f1
        if result_F1 < f1:
            result_Recall = recall
            result_F1 = f1
            result_Pre = precision
            result_Acc = acc
        # test
        print('\n\nepoch{}------------test------------'.format(epoch))
        acc, recall, precision, f1 = test(prompt_model, test_dataloader)
        if recall > bestRecall:
            bestRecall = recall
        if acc > bestAcc:
            bestAcc = acc
        if precision > bestPre:
            bestPre = precision
        if f1 > bestF1:
            bestF1 = f1
        if result_F1 < f1:
            result_Recall = recall
            result_F1 = f1
            result_Pre = precision
            result_Acc = acc
    print("\n\n best acc:{}   recall:{}   precision:{}   f1:{}".format(bestAcc, bestRecall, bestPre, bestF1))
    print("\n\n result acc:{}   recall:{}   precision:{}   f1:{}".format(result_Acc, result_Recall, result_Pre,
                                                                         result_F1))

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
    path = merge_tokens(slice_file_path)
    data = json.load(open(json_file_path))
    Classification(data, args.model_name, path)