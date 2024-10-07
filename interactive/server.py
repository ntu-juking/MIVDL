from client import train, test,show_representation
from model import FEDModel
import copy
import sys
from graph_dataset import DataSet
import json
import numpy
from sklearn.model_selection import train_test_split

class FedPer:
    def __init__(self, args):
        self.args = args
        self.nn = FEDModel()
        self.nns = []
        for i in range(self.args.client):
            temp = copy.deepcopy(self.nn)
            temp.name = self.args.clients[i]
            self.nns.append(temp)

    def server(self):
        for t in range(self.args.r):
            print('round', t + 1, ':')
            # dispatch
            self.dispatch()
            # local updating
            self.client_update()
            # aggregation
            self.aggregation()

        return self.nn

    #
    def aggregation(self):
        s = 0
        for j in range(self.args.client):
            s += self.nns[j].len

        for v in self.nn.parameters():
            v.data.zero_()

        for j in range(self.args.client):
            cnt = 0
            for v1, v2 in zip(self.nn.parameters(), self.nns[j].parameters()):
                v1.data += v2.data * (self.nns[j].len / s)
                cnt += 1
                if cnt == 2 * (self.args.total - self.args.Kp):
                    break

    # 将旧的参数更改为新的参数
    def dispatch(self):
        for j in range(self.args.client):
            cnt = 0
            for old_params, new_params in zip(self.nns[j].parameters(), self.nn.parameters()):
                old_params.data = new_params.data.clone()
                cnt += 1
                if cnt == 2 * (self.args.total - self.args.Kp):
                    break

    # 获取数据  但是数据集部分有问题  目前来看是有重复数据的
    def get_tr_te(self,k):
        ds = ''
        features = []
        targets = []
        parts = ['train', 'valid', 'test']
        # for part in parts:
        #     # json_data_file = open(ds + part + '_GNNinput_graph.json')
        #     # get json file
        #     json_data_file = open('../graph_feature_extraction/SDV_After_GNN/datasets-line-ggnn.json')
        #     data = json.load(json_data_file)
        #     json_data_file.close()



        #     for d in data:
        #         features.append(d['graph_feature'])
        #         targets.append(d['target'])
        #     del data
        json_data_file = open('../processed_data/Devign-line-ggnn.json')
        data = json.load(json_data_file)
        json_data_file.close()
        for d in data:
            features.append(d['graph_feature'])
            targets.append(d['target'])
        X = numpy.array(features)
        Y = numpy.array(targets)
        # split dataset,
        train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2)

        print(train_X.shape, train_Y.shape, test_X.shape, test_Y.shape, numpy.sum(train_Y), numpy.sum(test_Y), sep='\t', file=sys.stderr, flush=True)
        return train_X, train_Y,test_X,  test_Y

    # 训练模型
    def client_update(self):  # update nn
        for k in range(self.args.client):
            train_x, train_y, _, _ = self.get_tr_te(k)
            self.dataset = DataSet(128, train_x.shape[1])
            for x, y in zip(train_x, train_y):
                if numpy.random.uniform() <= 0.1:
                    self.dataset.add_data_entry(x.tolist(), y.item(), 'valid')
                else:
                    self.dataset.add_data_entry(x.tolist(), y.item(), 'train')
            self.dataset.initialize_dataset(balance=True, output_buffer=sys.stderr)
            self.nns[k] = train(self.args, self.nns[k], self.dataset)


    #
    def global_test(self):

        for j in range(self.args.client):
            _, _,test_x, test_y= self.get_tr_te(j)

            if not hasattr(self, 'dataset'):
                raise ValueError('Train First!')
            self.dataset.clear_test_set()
            for t_x, t_y in zip(test_x, test_y):
                self.dataset.add_data_entry(t_x.tolist(), t_y.item(), part='test')
            model = self.nns[j]
            model.eval()
            test(model, self.dataset)

