import torch
import pickle
import json
import argparse
from modules.model import GAT

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='path to GGNNSum model to be used', default='/opt/data/Devign/models/ffmqemDevign-ffmqem3000-model.bin')
parser.add_argument('--dataset', help='path to data to be processed (i.e. processed.bin)', default='/opt/data/ReVeal-master/CPVD_GNN_INPUT/github/github_4J_GAT_20000.bin')
parser.add_argument('--output_dir', help='location to place data after ggnn processing', default='./SDV_After_GNN/')
parser.add_argument('--name', help='name of folder to save data in (to differentiate sets)', default='devign_github')

args = parser.parse_args()

dataset = pickle.load(open(args.dataset, 'rb'))

state_dict = torch.load(args.model)

_model = GAT(input_dim=100, output_dim=100, num_steps=8,num_heads=2)

_model.load_state_dict(state_dict, strict=False)
_model.eval()
print('Data & Models Loaded')
print('='*83)


if dataset.test_batches:
    final = []
    for l in range(len(dataset.test_batches)):
        if dataset.inf:
            graph, targets, file_names = dataset.get_next_test_batch() 
        else:
            graph, targets = dataset.get_next_test_batch()


        output = _model.get_graph_embeddings(graph)
        if dataset.inf:
            final += list(zip(output.tolist(), [int(i) for i in targets.tolist()], file_names))
        else:
            final += list(zip(output.tolist(), [int(i) for i in targets.tolist()]))

    out = []
    if dataset.inf:
        for f in final:
            out.append({'graph_feature':f[0], 'target':f[1], 'file_name':f[2]})
    else:
        for f in final:
            out.append({'graph_feature':f[0], 'target':f[1]})

    # with open(args.output_dir + args.name + '/test_GNNinput_graph.json', 'w') as of:
    with open(args.output_dir + '/datasets-line-ggnn.json', 'w') as of:
        json.dump(out, of, indent=2)
        of.close()

    print('DONE: TEST BATCHES')


if dataset.valid_batches:
    final = []
    for l in range(len(dataset.valid_batches)):
        if dataset.inf:
            graph, targets, file_names = dataset.get_next_valid_batch() 
        else:
            graph, targets = dataset.get_next_valid_batch()


        output = _model.get_graph_embeddings(graph)
        if dataset.inf:
            final += list(zip(output.tolist(), [int(i) for i in targets.tolist()], file_names))
        else:
            final += list(zip(output.tolist(), [int(i) for i in targets.tolist()]))

    out = []
    if dataset.inf:
        for f in final:
            out.append({'graph_feature':f[0], 'target':f[1], 'file_name':f[2]})
    else:
        for f in final:
            out.append({'graph_feature':f[0], 'target':f[1]})

    # with open(args.output_dir + args.name + '/valid_GNNinput_graph.json', 'w') as of:
    with open(args.output_dir + '/datasets-line-ggnn.json', 'w') as of:
        json.dump(out, of, indent=2)
        of.close()

    print('DONE: VALID BATCHES')


if dataset.train_batches:
    final = []
    for l in range(len(dataset.train_batches)):
        if dataset.inf:
            graph, targets, file_names = dataset.get_next_train_batch() 
        else:
            graph, targets = dataset.get_next_train_batch()


        output = _model.get_graph_embeddings(graph)
        if dataset.inf:
            final += list(zip(output.tolist(), [int(i) for i in targets.tolist()], file_names))
        else:
            final += list(zip(output.tolist(), [int(i) for i in targets.tolist()]))

    out = []
    if dataset.inf:
        for f in final:
            out.append({'graph_feature':f[0], 'target':f[1], 'file_name':f[2]})
    else:
        for f in final:
            out.append({'graph_feature':f[0], 'target':f[1]})

    # with open(args.output_dir + args.name + '/train_GNNinput_graph.json', 'w') as of:
    with open(args.output_dir + '/datasets-line-ggnn.json', 'w') as of:
        json.dump(out, of, indent=2)
        of.close()

    print('DONE: TRAIN BATCHES')


print('='*100)
print('COMPLETED')
print('='*100)
