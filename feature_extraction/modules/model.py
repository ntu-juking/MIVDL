import torch
from dgl.nn.pytorch import GATConv
from torch import nn
import torch.nn.functional as f
import dgl

class GAT(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, num_steps=8):
        super(GAT, self).__init__()
        self.inp_dim = input_dim
        self.out_dim = output_dim
        self.num_heads = num_heads
        self.num_timesteps = num_steps
        self.gat1 = GATConv(in_feats=input_dim, out_feats=output_dim,num_heads=num_heads,allow_zero_in_degree=True)
        self.gat2 = GATConv(in_feats=output_dim * num_heads, out_feats=output_dim, num_heads=num_heads, allow_zero_in_degree=True)
        self.gat3 = GATConv(in_feats=output_dim * num_heads, out_feats=output_dim, num_heads=num_heads,allow_zero_in_degree=True)
        self.conv_l1 = torch.nn.Conv1d(200, 200, 3)
        self.maxpool1 = torch.nn.MaxPool1d(3, stride=2)
        self.conv_l2 = torch.nn.Conv1d(200, 200, 1)
        self.maxpool2 = torch.nn.MaxPool1d(2, stride=2)
        self.classifier = nn.Linear(in_features =200, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, batch, cuda=False):
        graph, features = batch.get_network_inputs(cuda=cuda)
        graph = dgl.add_self_loop(graph)
        features1 = self.gat1(graph, features)
        features1 = f.relu(features1)
        features1 = torch.flatten(features1, 1)
        features2 = self.gat2(graph, features1)
        features2 = f.relu(features2)
        features2 = torch.flatten(features2, 1)
        features3 = self.gat3(graph, features2)
        features3 = f.relu(features3)
        features3 = torch.flatten(features3, 1)
        features4 = self.gat3(graph, features3)
        features4 = f.relu(features4)
        features4 = torch.flatten(features4, 1)

        features5 = features1 + features2 + features3 + features4

        h_i, _ = batch.de_batchify_graphs(features5)

        Y_1 = self.maxpool1(f.relu(self.conv_l1(h_i.transpose(1, 2))))
        Y_2 = self.maxpool2(f.relu(self.conv_l2(Y_1))).transpose(1, 2)
        Y_2 = Y_2.sum(dim=1)
        avg = self.classifier(Y_2)
        result = self.sigmoid(avg).squeeze(dim=-1)
        return result

    def get_graph_embeddings(self, batch, cuda=False):
        graph, features = batch.get_network_inputs(cuda=cuda)
        graph = dgl.add_self_loop(graph)
        features1 = self.gat1(graph, features)
        features1 = f.relu(features1)
        features1 = torch.flatten(features1, 1)
        features2 = self.gat2(graph, features1)
        features2 = f.relu(features2)
        features2 = torch.flatten(features2, 1)
        features3 = self.gat3(graph, features2)
        features3 = f.relu(features3)
        features3 = torch.flatten(features3, 1)
        features4 = self.gat3(graph, features3)
        features4 = f.relu(features4)
        features4 = torch.flatten(features4, 1)

        features5 = features1 + features2 + features3 +features4
        h_i, _ = batch.de_batchify_graphs(features5)
        batch_size, num_node, _ = h_i.size()
        Y_1 = self.maxpool1(f.relu(self.conv_l1(h_i.transpose(1, 2))))
        Y_2 = self.maxpool2(f.relu(self.conv_l2(Y_1))).transpose(1, 2)
        Y_2 = Y_2.sum(dim=1)

        return Y_2.detach().cpu()