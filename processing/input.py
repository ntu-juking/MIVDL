import os
import json
import argparse
from tqdm import tqdm

def raw_code2dict(file_path):
    file_name = file_path.split('/')[-1]
    output = ""
    output = {
        'file_name':file_name,
        'label':int(file_name[-3]),
        'code':open(file_path, 'r', encoding='gbk', errors='ignore').read()
        }
    return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', help='name of project for differentiating files')
    parser.add_argument('--input', help='directory where raw code and parsed are stored', default='../data/chrome_debian')
    parser.add_argument('--c_source_path', help='raw_c_source_data', default='../raw_data/Devign/function.json')
    parser.add_argument('--output', help='output directory for resulting json file', default='../processed_data/ggnn_input/')
    args = parser.parse_args()

    #code_file_path = args.input + '/rawcode/'+args.project+'/'
    code_file_path = args.input
    output_data = []
    for cfile in tqdm(os.listdir(code_file_path)):
        fp = code_file_path + cfile
        output_data.append(raw_code2dict(fp))
    
    output_file = args.output + args.project + '_cpg_full_text_files.json'

    with open(output_file, 'w') as of:
        json.dump(output_data, of)
        of.close()
    print(f'Saved Output File to {output_file}')
