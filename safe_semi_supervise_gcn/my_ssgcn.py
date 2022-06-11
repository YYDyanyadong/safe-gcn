import os.path as osp
import argparse

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
# from torch_geometric.nn import GCNConv, ChebConv  # noqa
from MyApplication.safe_semi_supervise_gcn.my_gcn_conv import GCNConv

parser = argparse.ArgumentParser()
parser.add_argument('--use_gdc', action='store_true',
                    help='Use GDC preprocessing.')
args = parser.parse_args()
print('calling my_ssgcn')


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

    def forward(self, train_data, train_edge_index, training):
        x, edge_index, edge_weight = train_data, train_edge_index, None
        x = F.relu(self.conv1(x, edge_index, training, edge_weight))  # self.conv1会调用GCNConv中的forward()函数
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, training, edge_weight)
        # return F.log_softmax(x, dim=1)
        return x

# 在半监督gcn中，训练用到的A矩阵是通过整个train数据集得到的train_edge_index
def train(trainD, model, optimizer):
    train_data = trainD['train_data']
    train_label = trainD['train_label']
    train_edge_index = trainD['train_edge_index']
    train_mask_L = trainD['train_mask_L']

    model.train()
    optimizer.zero_grad()
    F.nll_loss(F.log_softmax(model(train_data, train_edge_index, True)[train_mask_L], dim=1), train_label[train_mask_L]).backward()
    optimizer.step()


# 在半监督gcn中，测试用到的A矩阵是也是通过整个train数据集得到的train_edge_index
@torch.no_grad()
def test_ssgcn(trainD, model):
    model.eval()

    train_mask_U = trainD['train_mask_U']
    train_label = trainD['train_label']
    train_data = trainD['train_data']
    train_edge_index = trainD['train_edge_index']
    logits = model(train_data, train_edge_index, False)
    pred = logits[train_mask_U].max(1)[1]
    acc = pred.eq(train_label[train_mask_U]).sum().item() / train_mask_U.sum().item()
    return acc

@torch.no_grad()
def test_ssgcn_on_test(test, model):
    model.eval()

    test_data = test['test_data']
    test_mask = test['test_mask']
    test_edge_index = test['test_edge_index']
    test_label = test['test_label']
    logits = model(test_data, test_edge_index, False)
    pred = logits.max(1)[1]
    acc = pred.eq(test_label).sum().item() / test_mask.sum().item()
    return acc

def acquire_logtis(dataset, edge_index, train_mask_U, model):
    logits = model(dataset, edge_index, False)  # 我把第三个参数由model改为了False
    logits = F.softmax(logits, dim=1)
    logits = logits[train_mask_U]   # 获取无标签样本的网络输出
    return logits


def ssgcnRun(trainD, dataIfo):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(dataIfo).to(device)
    optimizer = torch.optim.Adam([
        dict(params=model.reg_params, weight_decay=5e-4),
        dict(params=model.non_reg_params, weight_decay=0)
    ], lr=0.01)
    train_data = trainD['train_data']
    train_edge_index = trainD['train_edge_index']
    train_mask_U = trainD['train_mask_U']

    for epoch in range(1, 201):
        train(trainD, model, optimizer)
        train_acc = test_ssgcn(trainD, model)
        log = 'Epoch: {:03d}, Train: {:.4f}'
        # print(log.format(epoch, train_acc))

    return acquire_logtis(train_data, train_edge_index, train_mask_U, model)

def ssgcnRun_on_test(trainD, test, dataIfo):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(dataIfo).to(device)
    optimizer = torch.optim.Adam([
        dict(params=model.reg_params, weight_decay=5e-4),
        dict(params=model.non_reg_params, weight_decay=0)
    ], lr=0.01)
    train_data = trainD['train_data']
    train_edge_index = trainD['train_edge_index']
    train_mask_U = trainD['train_mask_U']

    for epoch in range(1, 201):
        train(trainD, model, optimizer)
        train_acc = test_ssgcn_on_test(test, model)
        log = 'Epoch: {:03d}, Train: {:.4f}'
        print(log.format(epoch, train_acc))

    return acquire_logtis(train_data, train_edge_index, train_mask_U, model)



