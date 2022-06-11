import os.path as osp
import argparse

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from MyApplication.safe_semi_supervise_gcn.dataset_op import chooseLabelOnFixNum, edgeIndexOnCut, edgeIndexOnSelect, edgeIndexOnIncrease, edgeIndexOnRange, chooseIndexOnFixNum2, acquire_index_label_U, update_index, edgeIndexNorm, update_index2
from MyApplication.safe_semi_supervise_gcn.my_sgcn import sgcnRun
from MyApplication.safe_semi_supervise_gcn.my_ssgcn import ssgcnRun
from MyApplication.safe_semi_supervise_gcn.my_ssgcn_split import ssgcnRun_split
from MyApplication.safe_semi_supervise_gcn.my_tagcn_split import tagcnRun_split
from MyApplication.safe_semi_supervise_gcn.my_sgc_split import sgcRun_split
import numpy as np

# 数据的下载与预处理
parser = argparse.ArgumentParser()
parser.add_argument('--use_gdc', action='store_true',
                    help='Use GDC preprocessing.')
args = parser.parse_args()

dataset_name = 'Cora'
dataset = dataset_name
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
data = dataset[0]

# 把数据放入到GPU中
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)

"""
数据集的划分
把总共2708个数据划分为train和test两个大的部分
其中train的数量为0:1899供1900个样本，test为1900:2707共808个样本
同时train又被划分为两个数据集，其中train_data_L为标记样本，数量一开始为0:190共190样本, train_data_U为未标记样本，数量一开始为190:1900共1710个
在模型的运行过程中我们会不断调整train_data_L和train_data_U中的数据集，其中的一种调整方式为每次迭代会把train_data_U中的一部分样本放入到train_data_L中，试验不同样本比例对模型的影响。
"""

"""
原始标签设置：
①Cora数据集共2708个数据，共7个类别，用0~6来标记
train样本范围0:139，共140个数据
val样本范围140:639，共500个数据
test样本范围1708:2707，共1000个数据
剩余2708-140-500-1000=1068个样本没有被使用

②Pubmed数据集共19717个数据，共3个类别，分别用0~2来标记。
train样本范围0:59，共60个数据
val样本范围60:559，共500个数据
test样本范围18171:19716，共1000个数据
剩余19717-140-500-1000=18077个样本没有被使用

③Citeseer数据集共3327个数据，共6个类别，分别用0~5来标记
train样本范围0:119，共120个数据
val样本范围120:619，共500个数据
test样本范围2312:3326，共1000个数据
剩余3327-120-500-1000=1707个样本没有被使用
"""
# class trainL:
#     traindata_L = torch.tensor([])
#     trainlabel_L = torch.tensor([])
#     trainedge_index_L = torch.tensor([])
#     trainindex_L = torch.tensor([])
# class dataIfo:
#     num_features = 0
#     num_classes = 0
#
# data_Ifo = dataIfo()
# data_Ifo.num_features = data.num_features
# data_Ifo.num_features = torch.max(data.y).item()+1


"""
参数设置
1.设定总的数据量:datanum
2.设定训练样本的数据量:trainnum
3.设定训练样本中有标签的数据量:trainnum_L
4.设定训练样本中无标签的数据量:trainnum_U
5.设定测试样本的数据量:testnum
"""


def dataset_choose(datasetname):
    if datasetname == 'Cora':
        datanum = 2708
        trainnum = 1708
        trainnum_L = 140
        trainnum_U = trainnum - trainnum_L
        testnum = datanum - trainnum
    elif datasetname == 'Citeseer':
        datanum = 3327
        trainnum = 2312
        trainnum_L = 120
        trainnum_U = trainnum - trainnum_L
        testnum = datanum - trainnum
    elif datasetname == 'Pubmed':
        datanum = 19717
        trainnum = 18170
        trainnum_L = 60
        trainnum_U = trainnum - trainnum_L
        testnum = datanum - trainnum
    return datanum, trainnum, trainnum_L, trainnum_U, testnum

data_num, train_num, train_num_L, train_num_U, test_num = dataset_choose(dataset_name)

# 获取样本集的基本信息,包括节点的特征向量和样本的类别总数
dataIfo = []
dataIfo.append(data.num_features)
dataIfo.append(torch.max(data.y).item()+1)

# 初始的样本集的下标设置，包括训练样本下标(train_index),训练样本中有标签的样本的下标(train_index_L),训练样本中无标签的样本的下标(train_index_U),以及测试样本的下标(test_index)
train_index = torch.linspace(0, train_num-1, train_num).to(device)
train_index_L = torch.linspace(0, train_num_L-1, train_num_L).to(device)
train_index_U = torch.linspace(train_num_L, train_num-1, train_num_U).to(device)
test_index = torch.linspace(train_num, data_num-1, test_num).to(device)

# 样本数据的副本(为了避免以直接引用的形式赋值)
datax = data.x.clone()
datay = data.y.clone()
datay[train_num_L:train_num] = -1     # 测试样本中无标签的类别都设置为-1
data_edge_index = data.edge_index.clone()

# 样本副本的划分
train_data = datax[0:train_num, :]
train_data_L = datax[train_index_L.long(), :]
train_data_U = datax[train_index_U.long(), :]
test_data = datax[test_index.long(), :]

# 标签副本的划分
train_label = datay[0:train_num]
train_label_L = datay[train_index_L.long()]
train_label_U = datay[train_index_U.long()]
test_label = datay[test_index.long()]


# 边集的划分，注意edgeindex的结束下标就是参数所示的数值。这里的data.edge_index是所有2708个样本数据的边坐标
train_edge_index = edgeIndexOnSelect(data_edge_index, train_index)
train_edge_index_L = edgeIndexOnSelect(data_edge_index, train_index_L)
train_edge_index_U = edgeIndexOnSelect(data_edge_index, train_index_U) - train_num_L
test_edge_index = edgeIndexOnSelect(data_edge_index, test_index) - train_num

# mask划分
# train的mask划分
train_mask_L = torch.zeros(train_num)
train_mask_L[0:train_num_L] = 1
train_mask_L = torch.eq(train_mask_L, 1)

train_mask_U = torch.zeros(train_num)
train_mask_U[train_num_L:train_num] = 1
train_mask_U = torch.eq(train_mask_U, 1)

# test的mask划分
test_mask = torch.zeros(data_num)
test_mask[train_num:data_num] = 1
test_mask = torch.eq(test_mask, 1)


# 数据的打包, 分别打包为train_L，train和train_U
train_L = []
train_L.append(train_data_L)
train_L.append(train_label_L)
train_L.append(train_edge_index_L)
train_L.append(train_index_L)

train = []
train.append(train_data)
train.append(train_label)
train.append(train_edge_index)
train.append(train_mask_L)
train.append(train_mask_U)
train.append(train_edge_index_U)
train.append(test_data)
train.append(test_label)
train.append(test_edge_index)


train_U = []
train_U.append(train_data_U)
train_U.append(train_label_U)
train_U.append(train_edge_index_U)
train_U.append(train_index_U)


# 更新数据，给所有train_L和train_U的数据重新编号，由于拉普拉斯矩阵计算的是每个节点与其它节点之间的关联，所以顺序的调整并不会影响后续的计算结果，只要节点之间依次对应就行。
# 在这个函数里面train_index_U1和train_index_L1的只是用于在data,label和edge_index获取数据，并不用在后续的操作中。
def data_refresh_data_index(train_index_U1, train_index_L1):
    # 利用choose_index对已经选择过的样本进行标记。

    # 由于U的下标是从190开始的，所以要重建U的下标的话应该减去190
    train_index_U1_values, train_index_U1_indices = torch.sort(train_index_U1-train_num_L).values, torch.sort(train_index_U1-190).indices
    train_index_L1_values, train_index_L1_indices = torch.sort(train_index_L1).values, torch.sort(train_index_L1).indices
    # 依据values和indices更新L和U中的数据、标签以及edge_index
    train_data_L = datax[train_index_L1.long(), :].clone()
    train_data_U = datax[train_index_U1.long(), :].clone()

    train_label_L = datay[train_index_L1.long()].clone()
    train_label_U = datay[train_index_U1.long()].clone()

    train_edge_index_L = edgeIndexOnSelect(data.edge_index, train_index_L1)
    train_edge_index_U = edgeIndexOnSelect(data.edge_index, train_index_U1) - train_num_L


    # 利用重建的下标来调整edge_index的范围，使其下标调整到当前样本集的范围之内。
    train_edge_index_L = edgeIndexNorm(train_edge_index_L, train_index_L1_values, train_index_L1_indices)  # 对edge_index的坐标范围进行转换
    train_edge_index_U = edgeIndexNorm(train_edge_index_U, train_index_U1_values, train_index_U1_indices)

    # 更新train_L， train_U和train中的参数
    train_L['train_data_L'] = train_data_L.clone()
    train_L['train_label_L'] = train_label_L.clone()
    train_L['train_edge_index_L'] = train_edge_index_L.clone()
    train_L['train_index_L'] = train_index_L1.clone()

    train_U['train_data_U'] = train_data_U.clone()
    train_U['train_label_U'] = train_label_U.clone()
    train_U['train_edge_index_U'] = train_edge_index_U.clone()
    train_U['train_index_U'] = train_index_U1.clone()

    # 转换坐标为mask, 使得train_mask_L中train_index_L1下标位置的布尔值为True，其余位置为False
    train_mask_L = torch.zeros(train_num)
    train_mask_L[train_index_L1.long()] = 1
    train_mask_L = torch.eq(train_mask_L, 1)

    train_mask_U = torch.zeros(train_num)
    train_mask_U[train_index_U1.long()] = 1
    train_mask_U = torch.eq(train_mask_U, 1)

    # train中数据的坐标都是对应于1900个train数据的。
    train['train_mask_L'] = train_mask_L.clone()
    train['train_mask_U'] = train_mask_U.clone()
    train['train_edge_index_U'] = train_edge_index_U.clone()


def s3_gcn(train_L, train_U, train):
    # 加载模型，获得train_U样本集在监督和半监督GCN上的输出
    # logits_sgcn_trian_U对应的就是train_U中的数据和下标，具体来说对应于train_data_U，train_label_U和train_edge_index_U
    # 但是为了与ssgcn进行比较，我们需要明确logits_sgcn_train_U对应于train数据集中的哪些数据，这样才能跟其做比较。
    # 通过ssgcn我们可以知道logits_ssgcn_train_U对应的坐标bool值是train[4]，也就是当前的train_mask_U
    logits_sgcn_train_U = sgcnRun(train_L, train_U, dataIfo)  # 有监督的gcn在训练的时候只用到了train_L，只有在测试的时候用到了train_U
    logits_ssgcn_train_U = ssgcnRun(train, dataIfo)  # 半监督的gcn在训练的时候用到了所有训练样本的数据和edge_index，在测试的时候只用到了train_U


    # 获取选中的元素的坐标，并以此更新train_index_U和train_index_L中的坐标已经对应的标签
    # train_index_U_choose的下标是相对于整个train数据而言，而choose_index_final的下标是局限在train_U的范围之内。
    choose_index_final, train_index_U_choose, label_new = acquire_index_label_U(logits_ssgcn_train_U, logits_sgcn_train_U, trainD['train_mask_U'], datay)

    train_index_U2, train_index_L2 = update_index2(train_index_U_choose, train_U['train_index_U'], train_L['train_index_L'])

    return train_index_U2, train_index_L2, choose_index_final, train_index_U_choose


def s3_gcn_run():
    # 这里对各个图卷积算法进行训练和测试，其中有sgc,tag等多种方法的对比实验
    # 这里的算法函数都加了一个split后缀，意思是把数据与集分成了训练和测试集。在训练时，由训练数据集构建相似性矩阵A，在测试时，由测试数据集构建相似性矩阵A'
    sgcRun_split(train, dataIfo)
    print('above is the result of sgcRun_separate')
    tagcnRun_split(train, dataIfo)
    print('above is the result of tagcnRun_separate')
    ssgcnRun_split(train, dataIfo)  # ssgcn是半监督gcn的意思，其实就是原始的GCN，只是在训练和测试的时候分别由训练数据集和测试数据集来构建相似性矩阵A
    print('above is the result of ssgcnRun_separate')
    train_index_U1, train_index_L1, choose_index, train_index_U_choose= s3_gcn(train_L, train_U, train)
    # 第一次运算完以后，对于sgcn(监督GCN)我们重新组织train_L和train_U中的数据
    # 因为以及安全半监督的框架，train_L中大概率会增加新的数据，这部分数据是通过框架由train_U而来，数据获取了一个伪标签。同时train_U中的数据会减少
    # 在sgcn中，每一次样本选择迭代完成之后会在再次训练时用新的train_L来构建相似性矩阵A，同时也会用train_U来构建测试时会用到的相似性矩阵A
    # 同时data和label数据不涉及到下标的修改，但是edge_index涉及到下标要调整到新的数据集(train_L增加后的数据集)大小以内，所以需要转换
    # 对于ssgcn(半监督GCN，也就是原始的GCN)，我们需要调整的是train中的中间三个数据，前三个数据是固定不变的，不需要调整
    # 这三个数据为train_mask_L,train_mask_U和train_edge_index_U，并且下标的范围对应于整个train数据集(大小1900，因为版监督的GCN用整个训练集来构建相似性矩阵A
    # ..............................................................................................................................................................................................*)
    data_refresh_data_index(train_index_U1, train_index_L1)
    iter = 1
    while(len(choose_index)!= 0):
        train_index_U1, train_index_L1, choose_index, train_index_U_choose = s3_gcn(train_L, train_U, train)
        data_refresh_data_index(train_index_U1, train_index_L1)
        iter = iter+1
        print('iter = :', iter)
        print('add', len(train_index_U_choose), 'unlabel data to labeled data')
        print('number of labeled data is ', len(train_index_L1))
        print('number of unlabel data is ', len(train_index_U1))
    print('iter = :', iter)
    print('length of train_index_U1:', len(train_index_U1))

    # 如果所有的样本都不再满足
    logits_ssgcn_train_U = ssgcnRun(train, dataIfo)
    print('the length of logits_ssgcn_train_U:', logits_ssgcn_train_U.size(0))
    print('the length of Unlabel', torch.sum(train['train_mask_U']))

    # 把train_U种剩下的无法确定的样本利用ssgcn来预测其样本种类并作为L加入到train的样本标签种,其中要做一次坐标转换，让整体数据样本由1900转换到2708
    rest_U_flag = 1  # 当rest_U_flag = 1时表明会让剩下的样本的标签利用ssgcn预测获取，rest_U_flag = 0表示丢弃这些剩下的无标签样本
    if rest_U_flag == 1:

        maxlabel_ssgcn = logits_ssgcn_train_U.max(1)[1]
        datay[torch.cat([train['train_mask_U'], torch.eq(torch.zeros(data_num-train_num), 1)])] = maxlabel_ssgcn

    # 到此只更新了label，但是相应的其它数据都需要调整。
    # 对监督GCN中的参数进行调整
        train_L['train_data_L'] = datax[0:train_num, :]
        train_L['train_label_L'] = datay[0:train_num]
        train_L['train_edge_index_L'] = train_edge_index
        train_L['train_index_L'] = torch.linspace(0, train_num-1, train_num)
    # 对半监督GCN中的参数进行调整
    elif rest_U_flag == 0:
        train_L['train_data_L'] = datax[train_index_L1.long(), :]
        train_L['train_label_L'] = datay[train_index_L1.long()]
        train_L['train_edge_index_L'] = edgeIndexOnSelect(data_edge_index, train_index_L1.long())
        train_L['train_index_L'] = train_index_L1

    train['train_mask_U'] = test_mask
    train['train_edge_index_U'] = test_edge_index

    # 利用监督的GCN对test数据集进行预测
    # 对test数据集进行打包
    test = []
    test.append(test_data)
    test.append(test_label)
    test.append(test_edge_index)


    print('the length of train_data_L', len(train_L['train_label_L']))
    print('the lenght of train_mask_L:', torch.sum(train['train_mask_L']))
    print('the cal-rate of test is:')
    logits_sgcn_test_final = sgcnRun(train_L, test, dataIfo)
    # logits_sgcn_test_final = sgcnRun2(train_L, test, dataIfo)

    # logits_ssgcn_train_U = ssgcnRun(train, dataIfo)
    # 利用半监督的GCN对test数据集进行预测，这里的训练数据只用到前190个，测试的数据test中的数据



s3_gcn_run()

# 如果只用train_L中的样本，通过监督的GCN来对test中的数据来进行分类
def Only_on_testdata_sgcn():
    test = []
    test.append(test_data)
    test.append(test_label)
    test.append(test_edge_index)

    sgcnRun(train_L, test, dataIfo)

# Only_on_testdata_sgcn()

# def Only_on_testdata_ssgcn():
#     train190_test = []
#     train190_test.append(train_data)
#     train190_test.append(train_label)
#     train190_test.append(train_edge_index)
#     train190_test.append(train_mask_L)
#
#     test_mask = torch.zeros(2708)
#     test_mask[190:train_num_U + 190] = 1
#     train_mask_U = torch.eq(train_mask_U, 1)
#
#     logits_ssgcn_train_U = ssgcnRun(train)





