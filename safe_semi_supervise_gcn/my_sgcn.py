import os.path as osp
import argparse

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from MyApplication.safe_semi_supervise_gcn.my_gcn_conv import GCNConv # noqa
# from torch_geometric.nn import GCNConv, ChebConv  # noqa
from MyApplication.safe_semi_supervise_gcn.dataset_op import edgeIndexOnCut, edgeIndexOnRange

parser = argparse.ArgumentParser()
parser.add_argument('--use_gdc', action='store_true',
                    help='Use GDC preprocessing.')
args = parser.parse_args()



class Net(torch.nn.Module):
    def __init__(self, dataIfo):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataIfo[0], 16, cached=True,
                             normalize=not args.use_gdc)
        self.conv2 = GCNConv(16, dataIfo[1], cached=True,
                             normalize=not args.use_gdc)
        # self.conv1 = ChebConv(data.num_features, 16, K=2)
        # self.conv2 = ChebConv(16, data.num_features, K=2)

        self.reg_params = self.conv1.parameters()
        self.non_reg_params = self.conv2.parameters()

    def forward(self, dataset, edge_index):
        x, edge_index, edge_weight = dataset, edge_index, None
        x = F.relu(self.conv1(x, edge_index, True, edge_weight))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, True, edge_weight)
        return F.log_softmax(x, dim=1)

def model_test(data, model):
    conv1_out = torch.matmul(data, model.conv1.weight) + model.conv1.bias
    conv2_in = F.relu(conv1_out)
    conv2_out = torch.matmul(conv2_in, model.conv2.weight) + model.conv2.bias
    return F.log_softmax(conv2_out, dim=1)

def model_test_onGraph(x, edge_index, model):
    edge_weight = None
    conv1_out = F.relu(model.conv1(x, edge_index, False, edge_weight))
    conv2_in = conv1_out
    conv2_out = model.conv2(conv2_in, edge_index, False, edge_weight)
    # return F.log_softmax(conv2_out, dim=1)
    return F.softmax(conv2_out, dim=1)

def acc_single(pred, label):
    acc = pred.eq(label).sum().item() / label.shape[0]
    return acc

def acc_set(datalist, labellist, model):
    accs = []
    for set, label in zip(datalist, labellist):
        logits_supervisor = model_test(set, model)
        pred_supervisor = logits_supervisor.max(1)[1]
        acc = pred_supervisor.eq(label).sum().item() / label.shape[0]
        accs.append(acc)
    return accs

def acc_set_onGraph(data, label, edge_index, model):
    logits_supervisor = model_test_onGraph(data, edge_index, model)
    pred_supervisor = logits_supervisor.max(1)[1]
    acc = pred_supervisor.eq(label).sum().item() / label.shape[0]
    return acc


def train(dataset, label, edge_index, optimizer, model):
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model(dataset, edge_index), label).backward()
    optimizer.step()




@torch.no_grad()
def test_onGraph(data_list, label_list, edge_index_list, model):
    model.eval()
    accs = acc_set_onGraph(data_list, label_list, edge_index_list, model)
    return accs


def acquire_logits_onGraph(dataset, edge_index, model):
    logits_supervisor_onGraph = model_test_onGraph(dataset, edge_index, model)
    return logits_supervisor_onGraph

def sgcnRun(train_L, train_U, dataIfo):
    train_dataset_L = train_L['train_data_L']
    train_label_L = train_L['train_label_L']
    train_edge_index_L = train_L['train_edge_index_L']

    train_dataset_U = train_U['train_data_U']
    train_label_U = train_U['train_label_U']
    train_edge_index_U = train_U['train_edge_index_U']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(dataIfo).to(device)
    optimizer = torch.optim.Adam([
        dict(params=model.reg_params, weight_decay=5e-4),
        dict(params=model.non_reg_params, weight_decay=0)
    ], lr=0.01)

    for epoch in range(1, 201):
        train(train_dataset_L, train_label_L, train_edge_index_L, optimizer, model)
        train_acc_U = test_onGraph(train_dataset_U, train_label_U, train_edge_index_U, model)

        log = 'Epoch: {:03d}, Train_U: {:.4f}'
        # print(log.format(epoch, train_acc_U))
    return acquire_logits_onGraph(train_dataset_U, train_edge_index_U, model)

def sgcnRun_on_test(train_L, test, dataIfo):
    train_dataset_L = train_L['train_data_L']
    train_label_L = train_L['train_label_L']
    train_edge_index_L = train_L['train_edge_index_L']

    test_data = test['test_data']
    test_label = test['test_label']
    test_edge_index = test['test_edge_index']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(dataIfo).to(device)
    optimizer = torch.optim.Adam([
        dict(params=model.reg_params, weight_decay=5e-4),
        dict(params=model.non_reg_params, weight_decay=0)
    ], lr=0.01)

    for epoch in range(1, 201):
        train(train_dataset_L, train_label_L, train_edge_index_L, optimizer, model)
        train_acc_U = test_onGraph(test_data, test_label, test_edge_index, model)

        log = 'Epoch: {:03d}, Train_U: {:.4f}'
    print(log.format(epoch, train_acc_U))
    return acquire_logits_onGraph(test_data, test_edge_index, model)

