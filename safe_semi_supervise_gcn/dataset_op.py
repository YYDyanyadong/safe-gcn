import torch
import numpy as np

import os.path as osp
import argparse

from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from MyApplication.safe_semi_supervise_gcn.gcn_conv_22 import GCNConv # noqa

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def getDataIdx(cora_data):
    label_true = torch.tensor(1)
    label_true = label_true.bool()
    label_true_cuda = label_true.cuda()
    data = cora_data
    label_true_idx = (data == label_true_cuda).nonzero()
    return label_true_idx

def dataIndexOnEdge(edge_index, dataset_index, choose):
    dataset_index_max_onEdge = torch.eq(edge_index[0, :], dataset_index).nonzero()
    if choose == "max":
        dataset_index_end_onEdge = torch.max(dataset_index_max_onEdge, 0)
        return  dataset_index_end_onEdge[0].item()
    else:
        dataset_index_start_onEdge = torch.min(dataset_index_max_onEdge, 0)
        return dataset_index_start_onEdge[0].item()

def edgeIndexOnCut(full_edge_index, cut_index):
    edge_index_OnCut = full_edge_index[:, 0:dataIndexOnEdge(full_edge_index, cut_index, "max")+1]
    edge_index_OnCut = edge_index_OnCut[:, edge_index_OnCut[1, :] <= cut_index]
    return edge_index_OnCut

def edgeIndexOnIncrease(full_edge_index, edge_index, increase_num):
    end_index = edge_index[0, -1].item()
    edge_index = edgeIndexOnRange(full_edge_index, 0, end_index+increase_num)
    return edge_index

def edgeIndexOnSelect(full_edge_index, index_list):
    ups = torch.zeros(len(full_edge_index[1])).to(device)
    ups = torch.eq(ups, 1)
    downs = ups
    for i in index_list:
        up = torch.eq(full_edge_index[0, :], i)
        down = torch.eq(full_edge_index[1, :], i)
        ups = up | ups
        downs = down | downs
        index_bool = (ups & downs)
    edge_index = full_edge_index[:, index_bool]
    return edge_index



def edgeIndexOnRange(full_edge_index, index_start, index_end):
    edge_index_on_range = full_edge_index[:, dataIndexOnEdge(full_edge_index, index_start, "min"):(dataIndexOnEdge(full_edge_index, index_end, "max")+1)]
    edge_index_on_range = edge_index_on_range[:, index_start <= edge_index_on_range[1, :]]
    edge_index_on_range = edge_index_on_range[:, edge_index_on_range[1, :] <= index_end]
    return edge_index_on_range

def chooseLabelOnFixNum(data_label, hist, label_unique):
    acc_of_minLabel = hist.min(0)[0]
    label_indexs = torch.tensor([-1])
    for i in label_unique:
        label = torch.eq(data_label, i)
        label_index = torch.topk(label.int(), acc_of_minLabel.int())
        label_indexs = torch.cat((label_indexs, label_index.indices))
    label_indexs = label_indexs[1:]
    return label_indexs

def chooseIndexOnFixNum2(label_mix, hist, label_unique, value):
    acc_of_minLabel = torch.min(torch.histc(label_mix.float(), 7, 0, 6)[(torch.histc(label_mix.float(), 7, 0, 6) > 0)])
    label_indexs = torch.tensor([-1]).to(device)
    for i in label_unique:
        label_i = torch.eq(label_mix, i)
        label_val = value*label_i.int()
        hist = hist.int()
        label_index = torch.topk(label_i.int(), acc_of_minLabel.int())
        label_indexs = torch.cat((label_indexs, label_index.indices))
    label_indexs = label_indexs[1:]
    return label_indexs

def acquire_index_label_U(ssgcnlogits, sgcnlogits, train_mask_U, label_new):
    maxval_sgcn = sgcnlogits.max(1)[0]
    maxval_ssgcn = ssgcnlogits.max(1)[0]
    maxlabel_sgcn = sgcnlogits.max(1)[1]
    maxlabel_ssgcn = ssgcnlogits.max(1)[1]
    equal_label_bool = torch.eq(maxlabel_ssgcn, maxlabel_sgcn)
    bigthan_label_bool = torch.ge(maxval_ssgcn, maxval_sgcn)
    # bigthan_value_bool = torch.ge(maxval_ssgcn, 0.9)
    choose_label_bool = equal_label_bool & bigthan_label_bool #& bigthan_value_bool
    choose_index = torch.nonzero(choose_label_bool, as_tuple=False).squeeze(1)

    maxlabel_mix = maxlabel_ssgcn[choose_label_bool]
    maxss = maxval_ssgcn.clone()
    maxss = maxss[choose_label_bool]

    label_unique = torch.unique(maxlabel_mix)
    num_label = len(label_unique)
    print('the number of classes:', num_label)
    if len(maxlabel_mix) != 0:
        hist = torch.histc(maxlabel_mix.float(), num_label)
        choose_label_on_avg = chooseIndexOnFixNum2(maxlabel_mix, hist, label_unique, maxss)
        choose_label_on_avg = torch.sort(choose_label_on_avg)
        choose_index_final = choose_index[choose_label_on_avg.values]
        train_index_U = torch.nonzero(train_mask_U, as_tuple=False)
        train_index_U_choose = train_index_U[choose_index_final].squeeze(1)

        label_new[train_index_U_choose] = maxlabel_ssgcn[choose_index_final]
        return choose_index_final, train_index_U_choose.to(device), label_new
    else:
        choose_index_final = torch.tensor([])
        train_index_U_choose = torch.tensor([])
        return choose_index_final, train_index_U_choose.to(device), label_new

def update_index(choose_index_final, train_index_U, train_index_L):
    train_index_U1 = train_index_U.numpy()
    train_index_U1 = np.delete(train_index_U1, choose_index_final.numpy(), axis=0)
    train_index_U1 = torch.from_numpy(train_index_U1)
    train_index_L1 = torch.sort(torch.cat((train_index_L.int(), choose_index_final.int()+190))).values
    return train_index_U1, train_index_L1

def update_index2(train_index_U_choose, train_index_U, train_index_L):
    train_index_L1 = torch.sort(torch.cat((train_index_L.int(), train_index_U_choose.int()))).values
    train_index_U = train_index_U.cpu()
    train_index_U1 = train_index_U.numpy()
    train_index_U_choose = train_index_U_choose.cpu()
    train_index_U_choose = train_index_U_choose.numpy()
    train_index_U1 = np.setdiff1d(train_index_U1, train_index_U_choose)
    train_index_U1 = torch.from_numpy(train_index_U1)
    train_index_U1 = train_index_U1.to(device)
    return train_index_U1, train_index_L1

def edgeIndexNorm(edge_index, values, indices):
    boolvalue1 = torch.eq(edge_index[0], torch.unsqueeze(values, 1))
    boolvalue2 = torch.eq(edge_index[1], torch.unsqueeze(values, 1))
    for i in range(0, len(indices)):
        edge_index[0][boolvalue1[i]] = indices[i]
        edge_index[1][boolvalue2[i]] = indices[i]
    return edge_index



