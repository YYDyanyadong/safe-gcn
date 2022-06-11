import os.path as osp
import argparse

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from MyApplication.safe_semi_supervise_gcn.dataset_op import chooseLabelOnFixNum, edgeIndexOnCut, edgeIndexOnSelect, edgeIndexOnIncrease, edgeIndexOnRange, chooseIndexOnFixNum2, acquire_index_label_U, update_index, edgeIndexNorm, update_index2
from MyApplication.safe_semi_supervise_gcn.my_sgcn import sgcnRun, sgcnRun_on_test
from MyApplication.safe_semi_supervise_gcn.my_ssgcn import ssgcnRun, ssgcnRun_on_test
from MyApplication.safe_semi_supervise_gcn.my_ssgcn_split import ssgcnRun_split
from MyApplication.safe_semi_supervise_gcn.my_tagcn_split import tagcnRun_split
from MyApplication.safe_semi_supervise_gcn.my_sgc_split import sgcRun_split
from MyApplication.safe_semi_supervise_gcn.gat_split import gatRun_split
from MyApplication.safe_semi_supervise_gcn.my_super_GAT_split import super_GAT_split
from MyApplication.safe_semi_supervise_gcn.APPNP import APPNP_split
import numpy as np


def init_seeds(seed):
    torch.manual_seed(seed) # sets the seed for generating random numbers.
    torch.cuda.manual_seed(seed) # Sets the seed for generating random numbers for the current GPU. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
    torch.cuda.manual_seed_all(seed) # Sets the seed for generating random numbers on all GPUs. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.

    if seed == 0:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

init_seeds(0)

parser = argparse.ArgumentParser()
parser.add_argument('--use_gdc', action='store_true',
                    help='Use GDC preprocessing.')
args = parser.parse_args()

dataset_name = 'Citeseer'
dataset = dataset_name
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
data = dataset[0]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)

print('Running on the ', dataset_name)

def dataset_choose(datasetname):
    if datasetname == 'Cora':
        datanum = 2708
        trainnum = 1708
        trainnum_L = 140
        trainnum_U = trainnum - trainnum_L
        testnum = datanum - trainnum
    elif datasetname == 'Citeseer':
        datanum = 3327
        trainnum = 2327
        trainnum_L = 120
        trainnum_U = trainnum - trainnum_L
        testnum = datanum - trainnum
    elif datasetname == 'Pubmed':
        datanum = 19717
        trainnum = 18717
        trainnum_L = 60
        trainnum_U = trainnum - trainnum_L
        testnum = datanum - trainnum
    return datanum, trainnum, trainnum_L, trainnum_U, testnum

# data_num, train_num, train_num_L, train_num_U, test_num = dataset_choose(dataset_name)

def dataset_choose_on_ratio(datasetname, ratio):
    if datasetname == 'Cora':
        datanum = 2708
        trainnum = 1708
        trainnum_L = 140+trainnum*ratio
        trainnum_U = trainnum - int(trainnum_L)
        testnum = datanum - trainnum
    elif datasetname == 'Citeseer':
        datanum = 3327
        trainnum = 2327
        trainnum_L = 120+trainnum*ratio
        trainnum_U = trainnum - int(trainnum_L)
        testnum = datanum - trainnum
    elif datasetname == 'Pubmed':
        datanum = 19717
        trainnum = 18717
        trainnum_L = 60+trainnum*ratio
        trainnum_U = trainnum - int(trainnum_L)
        testnum = datanum - trainnum
    return datanum, trainnum, int(trainnum_L), int(trainnum_U), testnum

_ratio = 0
data_num, train_num, train_num_L, train_num_U, test_num = dataset_choose_on_ratio(dataset_name, _ratio)

dataIfo = []
dataIfo.append(data.num_features)
dataIfo.append(torch.max(data.y).item()+1)

def set_index(datanum, trainnum, trainnum_L, trainnum_U, testnum):
    trainindex = torch.linspace(0, trainnum - 1, trainnum).to(device)
    trainindex_L = torch.linspace(0, trainnum_L - 1, trainnum_L).to(device)
    trainindex_U = torch.linspace(trainnum_L, trainnum - 1, trainnum_U).to(device)
    testindex = torch.linspace(trainnum, datanum - 1, testnum).to(device)
    return trainindex, trainindex_L, trainindex_U, testindex

train_index, train_index_L, train_index_U, test_index = \
    set_index(data_num, train_num, train_num_L, train_num_U, test_num)

datax = data.x.clone()
datay = data.y.clone()
datay[train_num_L:train_num] = -1
data_edge_index = data.edge_index.clone()

train_data = datax[0:train_num, :]
train_data_L = datax[train_index_L.long(), :]
train_data_U = datax[train_index_U.long(), :]
test_data = datax[test_index.long(), :]

train_label = datay[0:train_num]
train_label_L = datay[train_index_L.long()]
train_label_U = datay[train_index_U.long()]
test_label = datay[test_index.long()]

train_edge_index = edgeIndexOnSelect(data_edge_index, train_index)
train_edge_index_L = edgeIndexOnSelect(data_edge_index, train_index_L)
train_edge_index_U = edgeIndexOnSelect(data_edge_index, train_index_U) - train_num_L
test_edge_index = edgeIndexOnSelect(data_edge_index, test_index) - train_num

train_mask_L = torch.zeros(train_num)
train_mask_L[0:train_num_L] = 1
train_mask_L = torch.eq(train_mask_L, 1)

train_mask_U = torch.zeros(train_num)
train_mask_U[train_num_L:train_num] = 1
train_mask_U = torch.eq(train_mask_U, 1)

test_mask = torch.zeros(data_num)
test_mask[train_num:data_num] = 1
test_mask = torch.eq(test_mask, 1)


train_L = {'train_data_L': train_data_L, 'train_label_L': train_label_L, \
           'train_edge_index_L': train_edge_index_L, 'train_index_L': train_index_L}
train = {'train_data': train_data, 'train_label': train_label, \
         'train_edge_index': train_edge_index, 'train_mask_L': train_mask_L, \
         'train_mask_U': train_mask_U, 'train_edge_index_U': train_edge_index_U, \
         'test_data': test_data, 'test_label': test_label, \
         'test_edge_index': test_edge_index, 'test_mask': test_mask}
train_U = {'train_data_U': train_data_U, 'train_label_U': train_label_U, \
           'train_edge_index_U': train_edge_index_U, 'train_index_U': train_index_U}
test = {'test_data': test_data, 'test_label': test_label, \
        'test_edge_index': test_edge_index, 'test_mask': test_mask}


def data_refresh_data_index(train_index_U1, train_index_L1):
    train_index_U1_values, train_index_U1_indices = torch.sort(train_index_U1 - train_num_L).values, torch.sort(train_index_U1 - train_num_L).indices
    train_index_L1_values, train_index_L1_indices = torch.sort(train_index_L1).values, torch.sort(train_index_L1).indices

    train_data_L = datax[train_index_L1.long(), :].clone()
    train_data_U = datax[train_index_U1.long(), :].clone()

    train_label_L = datay[train_index_L1.long()].clone()
    train_label_U = datay[train_index_U1.long()].clone()

    train_edge_index_L = edgeIndexOnSelect(data.edge_index, train_index_L1)
    train_edge_index_U = edgeIndexOnSelect(data.edge_index, train_index_U1) - train_num_L


    train_edge_index_L = edgeIndexNorm(train_edge_index_L, train_index_L1_values, train_index_L1_indices)
    train_edge_index_U = edgeIndexNorm(train_edge_index_U, train_index_U1_values, train_index_U1_indices)

    train_L['train_data_L'] = train_data_L.clone()
    train_L['train_label_L'] = train_label_L.clone()
    train_L['train_edge_index_L'] = train_edge_index_L.clone()
    train_L['train_index_L'] = train_index_L1.clone()

    train_U['train_data_U'] = train_data_U.clone()
    train_U['train_label_U'] = train_label_U.clone()
    train_U['train_edge_index_U'] = train_edge_index_U.clone()
    train_U['train_index_U'] = train_index_U1.clone()

    train_mask_L = torch.zeros(train_num)
    train_mask_L[train_index_L1.long()] = 1
    train_mask_L = torch.eq(train_mask_L, 1)

    train_mask_U = torch.zeros(train_num)
    train_mask_U[train_index_U1.long()] = 1
    train_mask_U = torch.eq(train_mask_U, 1)

    train['train_mask_L'] = train_mask_L.clone()
    train['train_mask_U'] = train_mask_U.clone()
    train['train_edge_index_U'] = train_edge_index_U.clone()


def s3_gcn(train_L, train_U, train):
    logits_sgcn_train_U = sgcnRun(train_L, train_U, dataIfo)
    logits_ssgcn_train_U = ssgcnRun(train, dataIfo)

    choose_index_final, train_index_U_choose, label_new = acquire_index_label_U(logits_ssgcn_train_U, logits_sgcn_train_U, train['train_mask_U'], datay)
    train_index_U2, train_index_L2 = update_index2(train_index_U_choose, train_U['train_index_U'], train_L['train_index_L'])

    return train_index_U2, train_index_L2, choose_index_final, train_index_U_choose


def s3_gcn_run():
    init_seeds(0)
    gatRun_split(train, dataIfo)
    print('the result of gat')
    init_seeds(0)
    tagcnRun_split(train, dataIfo)
    print('the result of tagcnRun_separate')
    init_seeds(0)
    sgcnRun_on_test(train_L, test, dataIfo)
    print('the result of sgcnRun_separate')
    init_seeds(0)
    ssgcnRun_split(train, dataIfo)
    print('the result of ssgcnRun_separate')
    init_seeds(0)
    train_index_U1, train_index_L1, choose_index, train_index_U_choose= s3_gcn(train_L, train_U, train)

    data_refresh_data_index(train_index_U1, train_index_L1)
    iter = 1
    while(len(choose_index)!= 0):
        train_index_U1, train_index_L1, choose_index, train_index_U_choose = s3_gcn(train_L, train_U, train)
        data_refresh_data_index(train_index_U1, train_index_L1)
        iter = iter+1
        print('iter = :', iter)
        print('from labeled dataset', 'apeend', len(train_index_U_choose), 'data')
        print('the number of labeled data ', len(train_index_L1))
        print('the number of unlabeled data ', len(train_index_U1))
    print('iter = :', iter)
    print('The number of unlabeled data in the train dataset:', len(train_index_U1))

    logits_ssgcn_train_U = ssgcnRun(train, dataIfo)
    logits_sgcn_train_U = sgcnRun(train_L, train_U, dataIfo)

    maxlabel_sgcn = logits_sgcn_train_U.max(1)[1]
    maxlabel_ssgcn = logits_ssgcn_train_U.max(1)[1]
    equal_label_bool = torch.eq(maxlabel_ssgcn, maxlabel_sgcn)
    rest_U_flag = 0
    if rest_U_flag == 1:
        maxlabel_ssgcn = logits_ssgcn_train_U.max(1)[1]
        datay[torch.cat([train['train_mask_U'], torch.eq(torch.zeros(data_num-train_num), 1)])] = maxlabel_ssgcn

        train_L['train_data_L'] = datax[0:train_num, :]
        train_L['train_label_L'] = datay[0:train_num]
        train_L['train_edge_index_L'] = train_edge_index
        train_L['train_index_L'] = torch.linspace(0, train_num-1, train_num)
        train['train_label'] = datay[0:train_num]


    print('the length of train_data_L', len(train_L['train_label_L']))
    # print('the lenght of train_mask_L:', torch.sum(train['train_mask_L']))
    print('the cal-rate of test on sgcn is:')
    logits_sgcn_test_final = sgcnRun_on_test(train_L, test, dataIfo)

    # logits_ssgcn_test_final = ssgcnRun_on_test(train, test, dataIfo)
    print('------------------------------------------------')

s3_gcn_run()





