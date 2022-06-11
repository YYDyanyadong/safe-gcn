# from sklearn import svm
# X = [[0, 0], [1, 1]]
# y = [0, 1]
# clf = svm.SVC(gamma='scale')
# clf.fit(X, y)
#
# print(clf.predict([[2., 2.]]))

from sklearn import svm
from sklearn.neural_network import MLPClassifier
import numpy as np
import os.path as osp
import argparse

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

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

# 把数据放入到GPU中 sklearn不能调用gpu
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# data = data.to(device)

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

# _ratio = 0  # 用来对初始的label与unlabel样本的比例进行调整
# data_num, train_num, train_num_L, train_num_U, test_num = dataset_choose_on_ratio(dataset_name, _ratio)

def set_index(datanum, trainnum, trainnum_L, trainnum_U, testnum):
    trainindex = torch.linspace(0, trainnum - 1, trainnum)
    trainindex_L = torch.linspace(0, trainnum_L - 1, trainnum_L)
    trainindex_U = torch.linspace(trainnum_L, trainnum - 1, trainnum_U)
    testindex = torch.linspace(trainnum, datanum - 1, testnum)
    return trainindex, trainindex_L, trainindex_U, testindex

# train_index, train_index_L, train_index_U, test_index = \
#     set_index(data_num, train_num, train_num_L, train_num_U, test_num)
#
# # 样本数据的副本(为了避免以直接引用的形式赋值)
# datax = data.x.clone()
# datay = data.y.clone()
# # datay[train_num_L:train_num] = -1     # 测试样本中无标签的类别都设置为-1
# data_edge_index = data.edge_index.clone()
#
# # 样本副本的划分
# train_data = datax[0:train_num, :]
# train_data_L = datax[train_index_L.long(), :]
# train_data_U = datax[train_index_U.long(), :]
# test_data = datax[test_index.long(), :]
#
# # 标签副本的划分
# train_label = datay[0:train_num]
# train_label_L = datay[train_index_L.long()]
# train_label_U = datay[train_index_U.long()]
# test_label = datay[test_index.long()]

# from sklearn import svm
# X = [[0, 0], [1, 1]]
# y = [0, 1]
# clf = svm.SVC(gamma='scale')
# clf.fit(X, y)
#
# print(clf.predict([[2., 2.]]))

from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
# X, y = make_classification(n_features=4, random_state=0)
# clf = make_pipeline(StandardScaler(),
#                     LinearSVC(random_state=0, tol=1e-5))
# clf.fit(train_data, train_label)
# ratio_ped = np.equal(clf.predict(test_data), test_label).sum().item()/1708
# print(ratio_ped)

# clf1 = svm.SVC(gamma='scale')
# clf1.fit(train_data_L, train_label_L)
# print(clf1.predict(test_data))
# # np.equal(clf1.predict(test_data), test_label).sum().item()/test_num
# ratio_ped1 = np.equal(clf1.predict(test_data), test_label).sum().item()/test_num
# print('SVM:', ratio_ped1)
#
#
# clf2 = MLPClassifier(random_state=1, max_iter=600).fit(train_data_L, train_label_L)
# ratio_ped2 = np.equal(clf2.predict(test_data), test_label).sum().item()/test_num
# print('BP:', ratio_ped2)

SVM_pre = []
BP_pre = []

for i in range(-3, 20, 1):
    if i < 11:
        _ratio = i / 1000
    else:
        _ratio = (i-9) / 100



    _ratio = 0
    # _ratio = i/100 # 用来对初始的label与unlabel样本的比例进行调整
    data_num, train_num, train_num_L, train_num_U, test_num = dataset_choose_on_ratio(dataset_name, _ratio)

    train_index, train_index_L, train_index_U, test_index = \
        set_index(data_num, train_num, train_num_L, train_num_U, test_num)

    # 样本数据的副本(为了避免以直接引用的形式赋值)
    datax = data.x.clone()
    datay = data.y.clone()
    # datay[train_num_L:train_num] = -1     # 测试样本中无标签的类别都设置为-1
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

    print('{0}{1}{2}'.format('--------------------', _ratio, '-----------------------'))
    clf1 = svm.SVC(gamma='scale')
    clf1.fit(train_data_L, train_label_L)
    # print(clf1.predict(test_data))
    # np.equal(clf1.predict(test_data), test_label).sum().item()/test_num
    ratio_ped1 = np.equal(clf1.predict(test_data), test_label).sum().item() / test_num
    print('SVM:', ratio_ped1)
    SVM_pre.append(ratio_ped1)

    clf2 = MLPClassifier(random_state=1, max_iter=1000).fit(train_data_L, train_label_L)
    ratio_ped2 = np.equal(clf2.predict(test_data), test_label).sum().item() / test_num
    print('BP:', ratio_ped2)
    BP_pre.append(ratio_ped2)

print(SVM_pre)
print(BP_pre)
