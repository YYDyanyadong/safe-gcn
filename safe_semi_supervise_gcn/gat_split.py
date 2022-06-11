import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
# from torch_geometric.nn import GATConv
from MyApplication.safe_semi_supervise_gcn.my_gat_conv import GATConv

# dataset = 'Cora'
# path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
# dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
# data = dataset[0]


class Net(torch.nn.Module):
    def __init__(self, dataIfo):
        super(Net, self).__init__()

        self.conv1 = GATConv(dataIfo[0], 8, heads=8, dropout=0.6)
        # On the Pubmed dataset, use heads=8 in conv2.
        # self.conv2 = GATConv(8 * 8, dataIfo[1], heads=8, concat=False, dropout=0.6)
        self.conv2 = GATConv(8 * 8, dataIfo[1], heads=1, concat=False,
                             dropout=0.6)

    def forward(self, train_data, train_edge_index):
        x, edge_index = train_data, train_edge_index
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=-1)



def train(trainD, model, optimizer):
    train_data = trainD['train_data']
    train_label = trainD['train_label']
    train_edge_index = trainD['train_edge_index']
    train_mask_L = trainD['train_mask_L']

    model.train()
    optimizer.zero_grad()
    F.nll_loss(model(train_data, train_edge_index)[train_mask_L], train_label[train_mask_L]).backward()
    optimizer.step()


def test_gat_split(trainD, model):
    model.eval()

    test_data = trainD['test_data']
    test_label = trainD['test_label']
    test_edge_index = trainD['test_edge_index']
    logits = model(test_data, test_edge_index)
    pred = logits.max(1)[1]
    acc = pred.eq(test_label).sum().item() / len(test_label)
    return acc


def gatRun_split(trainD, dataIfo):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(dataIfo).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    for epoch in range(1, 201):
        train(trainD, model, optimizer)
        train_acc = test_gat_split(trainD, model)
        log = 'Epoch: {:03d}, Train:{:.4f}'
    print(log.format(epoch, train_acc))



