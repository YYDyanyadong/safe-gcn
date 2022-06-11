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
        x = F.relu(self.conv1(x, edge_index, training, edge_weight))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, training, edge_weight)
        return F.log_softmax(x, dim=1)

# 训练模型
def train(trainD, model, optimizer):
    train_data = trainD['train_data']
    train_label = trainD['train_label']
    train_edge_index = trainD['train_edge_index']
    train_mask_L = trainD['train_mask_L']

    model.train()
    optimizer.zero_grad()
    F.nll_loss(model(train_data, train_edge_index, True)[train_mask_L], train_label[train_mask_L]).backward()
    optimizer.step()

@torch.no_grad()
def test_ssgcn_split(trainD, model):
    model.eval()

    test_data = trainD['test_data']
    test_label = trainD['test_label']
    test_edge_index = trainD['test_edge_index']
    logits = model(test_data, test_edge_index, False)
    pred = logits.max(1)[1]
    acc = pred.eq(test_label).sum().item() / len(test_label)
    return acc

def acquire_logtis_split(dataset, edge_index, model):
    logits = model(dataset, edge_index, False)
    logits = logits
    return logits


def ssgcnRun_split(trainD, dataIfo):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(dataIfo).to(device)
    optimizer = torch.optim.Adam([
        dict(params=model.reg_params, weight_decay=5e-4),
        dict(params=model.non_reg_params, weight_decay=0)
    ], lr=0.01)
    train_data = trainD['train_data']
    train_edge_index = trainD['train_edge_index']
    train_mask_U = trainD['train_mask_U']
    test_data = trainD['test_data']
    test_edge_index = trainD['test_edge_index']

    for epoch in range(1, 201):
        train(trainD, model, optimizer)
        train_acc = test_ssgcn_split(trainD, model)
        log = 'Epoch: {:03d}, Train: {:.4f}'
    print(log.format(epoch, train_acc))

    return acquire_logtis_split(test_data, test_edge_index, model)



