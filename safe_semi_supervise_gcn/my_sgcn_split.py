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



class Net(torch.nn.Module):   # 这个网络的精髓就在这里,具体的卷积过程在GCNConv里面
    def __init__(self, dataIfo): # 可以尝试在初始化上放入样本和标签的参数，这样在导出模型的时候就可以使用参数输入了。
        super(Net, self).__init__()   # 首先调用父类的构造函数，这样才能继承父类的属性
        self.conv1 = GCNConv(dataIfo[0], 16, cached=True,
                             normalize=not args.use_gdc)   # arg1是输入数据的通道数(每一个节点的特征维数，也就是样本的特征维数，在cora中共1433维。,arg2是输出数据的通道数(也就是样本通过卷积后映射的特征维数16)
        self.conv2 = GCNConv(16, dataIfo[1], cached=True,
                             normalize=not args.use_gdc)   # 对于cora数据来说总共是7类，从0~6类，所以dataset.num_classes=7
        # self.conv1 = ChebConv(data.num_features, 16, K=2)
        # self.conv2 = ChebConv(16, data.num_features, K=2)

        self.reg_params = self.conv1.parameters()    # 获得参数，如果这层网络有多个网络层构成，那么会返回多个网络参数
        self.non_reg_params = self.conv2.parameters()

    def forward(self, dataset, edge_index):   # 这里的forward除了self是没有参数的，因为数据集data已经在下一行传入了。
        # edge_index数据的shape=[2,n]，n为边数*2,第一行记录起点，第二行记录终点。其中n=10556所以边数目=5278,需要注意的是在cora中edge_weight=none，也就是边是没有权重的，只有两个节点之间存在或者不存在边。
        # 只获取前140个样本之间的关联无向图
        x, edge_index, edge_weight = dataset, edge_index, None   # 需要注意的是，在这个forward中已经把data数据传入其中，那么再次运行Net的对象比如model()或net()括号里面已经不需要传入数据集的参数，而是直接调用__call__()_，再通过调用forward。
        # x.shape=([2708,1433])  # 总 共有2708个数据点，每个数据点有1433维特征用来表示特殊词汇的存在与否。
        # 设置我自己的参数
        # 设置我自己的参数
        # conv1中的参数会传给GCNConv中的Forward()函数
        x = F.relu(self.conv1(x, edge_index, True, edge_weight))   # self.conv1(x, edge_index, edge_weight)又是以对象的形式调用方法，会触发__call__()，而该函数会调用GCNConv中的forward()函数，把参数传递给GCNConv中的forward()
        x = F.dropout(x, training=self.training)   # dropout也是正则化的一种
        x = self.conv2(x, edge_index, True, edge_weight)   # x的输出值是2708*7，两层卷积完成后x的维度从1433维降到了7维，且这个7与7个类别的Onehot编码的标签相对应。
        return F.log_softmax(x, dim=1)     # F.log_softmax(x, dim=1)得到的是聚集了各个节点信息和降维后的样本对应的特征向量。这里计算softmax的时候是通过特征向量一次性计算出其值(没有分步骤计算)，提高了效率和精度。

# 设备选择cuda，网络和数据都交给cuda
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = Net().to(device) # 把模型的数据和数据集的数据都放入到CUDA中, Net()是用默认参数生成Net对象。
# optimizer = torch.optim.Adam([
#     dict(params=model.reg_params, weight_decay=5e-4),
#     dict(params=model.non_reg_params, weight_decay=0)
# ], lr=0.01)

# 构建一个监督的GCN的样本测试计算，在测试过程中没有用到标准化后的拉普拉斯矩阵
def model_test(data, model):
    conv1_out = torch.matmul(data, model.conv1.weight) + model.conv1.bias
    conv2_in = F.relu(conv1_out)
    conv2_out = torch.matmul(conv2_in, model.conv2.weight) + model.conv2.bias
    # model.conv1(data,)
    return F.log_softmax(conv2_out, dim=1)
# 在测试过程中添加对应样本集的标准化邻接拉普拉斯矩阵
def model_test_onGraph(x, edge_index, model):
    edge_weight = None
    conv1_out = F.relu(model.conv1(x, edge_index, False, edge_weight))  # 第三个参数用来指示当前是在训练阶段还是预测阶段
    conv2_in = conv1_out
    conv2_out = model.conv2(conv2_in, edge_index, False, edge_weight)  # 第三个参数用来指示当前是在训练阶段还是预测阶段
    return F.log_softmax(conv2_out, dim=1)


# 对单个数据集计算准确率
def acc_single(pred, label):
    acc = pred.eq(label).sum().item() / label.shape[0]
    return acc

# 对多个数据集计算准确率
def acc_set(datalist, labellist, model):
    accs = []
    for set, label in zip(datalist, labellist):
        logits_supervisor = model_test(set, model)
        pred_supervisor = logits_supervisor.max(1)[1]
        acc = pred_supervisor.eq(label).sum().item() / label.shape[0]
        accs.append(acc)
    return accs

# 对多个数据集计算准确率(附上拉普拉斯矩阵)
def acc_setlist_onGraph(datalist, labellist, edge_index_list, model):
    accs = []
    for set, label, edge_index in zip(datalist, labellist, edge_index_list):
        logits_supervisor = model_test_onGraph(set, edge_index, model)
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
    # 以参数传递的形式直接修改data的内部数据,这样更方便，以免每次从data的内部获取数据，标签和边的信息。

    model.train()   # 启用 BatchNormalization 和 Dropout
    optimizer.zero_grad()  # 导数置零
    # 下一句的model()返回的是F.log_softmax(x, dim=1)
    # 需要注意的是，在这个forward中已经把data数据传入其中，那么再次运行Net的对象比如model()或net()括号里面已经不需要传入数据集的参数，而是直接调用__call__()_，再调用forward()。不过当第一次调用model()时首先调用的的还是__init__()函数，进行初始化.
    # data.train_mask.shape都等于2708，也就是说所有的数据都是有标签的，在训练f的过程中并没有用到半监督的概念，每一个train_mark中的值只有true和false。,其实true就代表有标签，false就代表没有标签。
    # model().shape的大小是[2708,7], data.y.shape=[2708],所以对于每一样本来说输出的log_softmax就是一个7维的特征向量，对应样本的标签是一个值的范围在0~6的标量。

    # data.y[data.train_mask]中当data.train_mask=1是表示使用其标签，否则不适用其标签，所以说这个训练是半监督训练，只用到了部分有标签的样本，而构建图模型的时候用到了所有的样本。
    # 获取model()返回的是tensor类型的log_softmax值，获取tensor中的值可以用索引的形式，就用索引位置的0,1来进行获取。
    # data.train_mask、data.test_mask和data.val_mask的size是相同的，只是在代表是否作为样本的bool值标注上有所不同。
    F.nll_loss(model(dataset, edge_index), label).backward()
    # F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward()    # nll_loss就是在负对数似然损失函数，当损失函数值是标量，那么在调用backward进行反向传播的梯度计算的时候就不需要传入参数。
    # F.nll_loss(model()[data.train_mask+data.test_mask+data.val_mask], data.y[data.train_mask+data.test_mask+data.val_mask]).backward()  # nll_loss就是在负对数似然损失函数，当损失函数值是标量，那么在调用backward进行反向传播的梯度计算的时候就不需要传入参数。
    optimizer.step()   # 只有用了optimizer.step()，模型才会更新




@torch.no_grad()
def test_onGraph(data_list, label_list, edge_index_list, model):
    model.eval()   # 不启用 BatchNormalization 和 Dropout，不计算梯度，因为这里只是在做预测
    accs = acc_set_onGraph(data_list, label_list, edge_index_list, model)
    return accs


def acquire_logits_onGraph(dataset, edge_index, model):
    logits_supervisor_onGraph = model_test_onGraph(dataset, edge_index, model)
    return logits_supervisor_onGraph



def sgcnRun(train_L, train_U, dataIfo):
    # 可以看到edge的坐标不需要进行全局调整，只需要调整到数据集的内部就行。
    # train_edge_index = edgeIndexOnCut(data.edge_index, 139)
    # val_edge_index = edgeIndexOnRange(data.edge_index, 140, 639) - 140   # 减去140是为了让坐标调整到数据集的内部，不至于下标超出数据集的大小。
    # test_edge_index = edgeIndexOnRange(data.edge_index, 1708, 2707) - 1708 # 减去1708是为了让坐标调整到数据集的内部，不至于下标超出数据集的大小。
    # edge_index_list = [train_edge_index, val_edge_index, test_edge_index]
    train_dataset_L = train_L['train_data_L']
    train_label_L = train_L['train_label_L']
    train_edge_index_L = train_L['train_edge_index_L']

    train_dataset_U = train_U['train_data_U']
    train_label_U = train_U['train_label_U']
    train_edge_index_U = train_U['train_edge_index_U']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(dataIfo).to(device)  # 把模型的数据和数据集的数据都放入到CUDA中, Net()是用默认参数生成Net对象。
    optimizer = torch.optim.Adam([
        dict(params=model.reg_params, weight_decay=5e-4),
        dict(params=model.non_reg_params, weight_decay=0)
    ], lr=0.01)

    for epoch in range(1, 201):  # 样本的训练，记录每一个epoch的train，val和test
        train(train_dataset_L, train_label_L, train_edge_index_L, optimizer, model)
        train_acc_U = test_onGraph(train_dataset_U, train_label_U, train_edge_index_U, model)  # 用已经训练好的监督GCN模型进行测试

        log = 'Epoch: {:03d}, Train_U: {:.4f}'
        print(log.format(epoch, train_acc_U))
    return acquire_logits_onGraph(train_dataset_U, train_edge_index_U, model)

def sgcnRun_on_test(train_L, test, dataIfo):
    # 可以看到edge的坐标不需要进行全局调整，只需要调整到数据集的内部就行。
    # train_edge_index = edgeIndexOnCut(data.edge_index, 139)
    # val_edge_index = edgeIndexOnRange(data.edge_index, 140, 639) - 140   # 减去140是为了让坐标调整到数据集的内部，不至于下标超出数据集的大小。
    # test_edge_index = edgeIndexOnRange(data.edge_index, 1708, 2707) - 1708 # 减去1708是为了让坐标调整到数据集的内部，不至于下标超出数据集的大小。
    # edge_index_list = [train_edge_index, val_edge_index, test_edge_index]
    train_dataset_L = train_L['train_data_L']
    train_label_L = train_L['train_label_L']
    train_edge_index_L = train_L['train_edge_index_L']

    test_data = test['test_data']
    test_label = test['test_label']
    test_edge_index = test['test_edge_index']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(dataIfo).to(device)  # 把模型的数据和数据集的数据都放入到CUDA中, Net()是用默认参数生成Net对象。
    optimizer = torch.optim.Adam([
        dict(params=model.reg_params, weight_decay=5e-4),
        dict(params=model.non_reg_params, weight_decay=0)
    ], lr=0.01)

    for epoch in range(1, 201):  # 样本的训练，记录每一个epoch的train，val和test
        train(train_dataset_L, train_label_L, train_edge_index_L, optimizer, model)
        train_acc_U = test_onGraph(test_data, test_label, test_edge_index, model)  # 用已经训练好的监督GCN模型进行测试

        log = 'Epoch: {:03d}, Train_U: {:.4f}'
        print(log.format(epoch, train_acc_U))
    return acquire_logits_onGraph(test_data, test_edge_index, model)

