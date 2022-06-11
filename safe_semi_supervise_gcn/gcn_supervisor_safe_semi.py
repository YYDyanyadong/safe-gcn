import os.path as osp
import argparse

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from MyApplication.safe_semi_supervise_gcn.gcn_conv_22 import GCNConv # noqa
# from torch_geometric.nn import GCNConv, ChebConv  # noqa
from MyApplication.safe_semi_supervise_gcn.dataset_op import edgeIndexOnCut, edgeIndexOnRange

parser = argparse.ArgumentParser()
parser.add_argument('--use_gdc', action='store_true',
                    help='Use GDC preprocessing.')
args = parser.parse_args()

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
data = dataset[0]

# 这里默认args.use_gdc=false,所以这里并没有进行GDC的处理。
if args.use_gdc:
    gdc = T.GDC(self_loop_weight=1, normalization_in='sym',
                normalization_out='col',
                diffusion_kwargs=dict(method='ppr', alpha=0.05),
                sparsification_kwargs=dict(method='topk', k=128,
                                           dim=0), exact=True)
    data = gdc(data)
# 在这里为止data里的数据并没有其变化。

# Cora数据说明
# data.x：节点特征矩阵，shape为[num_nodes, num_node_features]。
# data.edge_index：COO格式的graph connectivity矩阵，shape为[2, num_edges]，类型为torch.long。 行只有两维，其中第一行是边的起点坐标，第二行是边的终点坐标。
# data.edge_attr：边的特征矩阵，shape为[num_edges, num_edge_features]。边的特征不一定有值。
# data.y：训练的target，shape不固定，比如，对于node-level任务，形状为[num_nodes, *]，对于graph-level任务，形状为[1, *]。data.y就是标签
# data.pos：节点的位置(position)矩阵，shape为[num_nodes, num_dimensions]。
"""data.keys：返回属性名列表。
data['x']：返回属性名为'x'的值。
for key, item in data: ...：按照字典的方法返回data属性和对应值。
'x' in data：判断某一属性是否在data中。
data.num_nodes：返回节点个数，相当于edge_index.shape[1]。 shape[1]得到的是第二维的长度。
data.num_edges：返回边的条数，相当于x.shape[0]。 shape[0]得到的是第一维的长度
data.num_features在cora中是1433的长度，也就是说有每个样本点有1433维特征值，
data.contains_isolated_nodes()：是否存在孤立的节点。
data.contains_self_loops()：是否存在自环。
data.is_directed()：是否是有向图。
data.to(torch.device('cuda'))：将数据对象转移到GPU。"""

'''
节点数data.x=2708×1433
边数data.edge_index=10556=边数5278*2
data.edge_attr的特征为空
train的idx范围是0-139共140个，test的idx范围是140-639共500个样本，val的idx范围是1708-2707共1000个样本
'''



class Net(torch.nn.Module):   # 这个网络的精髓就在这里,具体的卷积过程在GCNConv里面
    def __init__(self): # 可以尝试在初始化上放入样本和标签的参数，这样在导出模型的时候就可以使用参数输入了。
        super(Net, self).__init__()   # 首先调用父类的构造函数，这样才能继承父类的属性
        self.conv1 = GCNConv(dataset.num_features, 16, cached=True,
                             normalize=not args.use_gdc)   # arg1是输入数据的通道数(每一个节点的特征维数，也就是样本的特征维数，在cora中共1433维。,arg2是输出数据的通道数(也就是样本通过卷积后映射的特征维数16)
        self.conv2 = GCNConv(16, dataset.num_classes, cached=True,
                             normalize=not args.use_gdc)   # 对于cora数据来说总共是7类，从0~6类，所以dataset.num_classes=7
        # self.conv1 = ChebConv(data.num_features, 16, K=2)
        # self.conv2 = ChebConv(16, data.num_features, K=2)

        self.reg_params = self.conv1.parameters()    # 获得参数，如果这层网络有多个网络层构成，那么会返回多个网络参数
        self.non_reg_params = self.conv2.parameters()

    def forward(self):   # 这里的forward除了self是没有参数的，因为数据集data已经在下一行传入了。
        # edge_index数据的shape=[2,n]，n为边数*2,第一行记录起点，第二行记录终点。其中n=10556所以边数目=5278,需要注意的是在cora中edge_weight=none，也就是边是没有权重的，只有两个节点之间存在或者不存在边。
        # 只获取前140个样本之间的关联无向图
        data.edge_index = edgeIndexOnCut(data.edge_index, 139)  # 由于后续的数据都是在data.edge_index上处理的，所以后续处理的edge_index是处理后的edge_index,这点一定要注意。
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr    # 需要注意的是，在这个forward中已经把data数据传入其中，那么再次运行Net的对象比如model()或net()括号里面已经不需要传入数据集的参数，而是直接调用__call__()_，再通过调用forward。
        # x.shape=([2708,1433])  # 总共有2708个数据点，每个数据点有1433维特征用来表示特殊词汇的存在与否。
        # 设置我自己的参数

        x = x[0:140, :]
        # 设置我自己的参数
        # edge_index = edgeIndexOnCut(edge_index1, 139)
        x = F.relu(self.conv1(x, edge_index, edge_weight))   # self.conv1(x, edge_index, edge_weight)又是以对象的形式调用方法，会触发__call__()，而该函数会调用GCNConv中的forward()函数，把参数传递给GCNConv中的forward()
        x = F.dropout(x, training=self.training)   # dropout也是正则化的一种
        x = self.conv2(x, edge_index, edge_weight)   # x的输出值是2708*7，两层卷积完成后x的维度从1433维降到了7维，且这个7与7个类别的Onehot编码的标签相对应。
        return F.log_softmax(x, dim=1)     # F.log_softmax(x, dim=1)得到的是聚集了各个节点信息和降维后的样本对应的特征向量。这里计算softmax的时候是通过特征向量一次性计算出其值(没有分步骤计算)，提高了效率和精度。

# 设备选择cuda，网络和数据都交给cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net().to(device), data.to(device)  # 把模型的数据和数据集的数据都放入到CUDA中, Net()是用默认参数生成Net对象。
optimizer = torch.optim.Adam([
    dict(params=model.reg_params, weight_decay=5e-4),
    dict(params=model.non_reg_params, weight_decay=0)
], lr=0.01)

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
    conv1_out = F.relu(model.conv1(x, edge_index, edge_weight))
    conv2_in = conv1_out
    conv2_out = model.conv2(conv2_in, edge_index, edge_weight)
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
def acc_set_onGraph(datalist, labellist, edge_index_list, model):
    accs = []
    for set, label, edge_index in zip(datalist, labellist, edge_index_list):
        logits_supervisor = model_test_onGraph(set, edge_index, model)
        pred_supervisor = logits_supervisor.max(1)[1]
        acc = pred_supervisor.eq(label).sum().item() / label.shape[0]
        accs.append(acc)
    return accs


def train():
    model.train()   # 启用 BatchNormalization 和 Dropout
    optimizer.zero_grad()  # 导数置零
    # 下一句的model()返回的是F.log_softmax(x, dim=1)
    # 需要注意的是，在这个forward中已经把data数据传入其中，那么再次运行Net的对象比如model()或net()括号里面已经不需要传入数据集的参数，而是直接调用__call__()_，再调用forward()。不过当第一次调用model()时首先调用的的还是__init__()函数，进行初始化.
    # data.train_mask.shape都等于2708，也就是说所有的数据都是有标签的，在训练的过程中并没有用到半监督的概念，每一个train_mark中的值只有true和false。,其实true就代表有标签，false就代表没有标签。
    # model().shape的大小是[2708,7], data.y.shape=[2708],所以对于每一样本来说输出的log_softmax就是一个7维的特征向量，对应样本的标签是一个值的范围在0~6的标量。

    # data.y[data.train_mask]中当data.train_mask=1是表示使用其标签，否则不适用其标签，所以说这个训练是半监督训练，只用到了部分有标签的样本，而构建图模型的时候用到了所有的样本。
    # 获取model()返回的是tensor类型的log_softmax值，获取tensor中的值可以用索引的形式，就用索引位置的0,1来进行获取。
    # data.train_mask、data.test_mask和data.val_mask的size是相同的，只是在代表是否作为样本的bool值标注上有所不同。
    # train的idx范围是0-139共140个，test的idx范围是140-639共500个样本，val的idx范围是1708-2707共1000个样本
    F.nll_loss(model(), data.y[0:140]).backward()
    # F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward()    # nll_loss就是在负对数似然损失函数，当损失函数值是标量，那么在调用backward进行反向传播的梯度计算的时候就不需要传入参数。
    # F.nll_loss(model()[data.train_mask+data.test_mask+data.val_mask], data.y[data.train_mask+data.test_mask+data.val_mask]).backward()  # nll_loss就是在负对数似然损失函数，当损失函数值是标量，那么在调用backward进行反向传播的梯度计算的时候就不需要传入参数。
    optimizer.step()   # 只有用了optimizer.step()，模型才会更新


@torch.no_grad()
def test(data_list, label_list):    # 在测试部分是不计算梯度的，所以如果需要改动模型并利用模型中已经学习好的参数那么就在test()里面改。
    # 可以考虑在test网络模型上加入两个参数：样本和标签，这样在保存再调用模型的时候就可以输入参数了。
    model.eval()   # 不启用 BatchNormalization 和 Dropout，不计算梯度，因为这里只是在做预测。

    # print(F.log_softmax(torch.matmul(F.relu(torch.matmul(data.x[140:639, :], model.conv1.weight)+model.conv1.bias), model.conv2.weight) + model.conv2.bias, dim=1))
    # model.conv1 = torch.matmul(torch.zeros(model.conv1.shape), model.conv1)  # 测试一下能否对model中的模型进行赋值
    logits, accs = model(), []    # 从model获取的logits已经通过softmax转换成概率向量。

    train_data = data.x[0:140, :]
    logtis1 = model_test(train_data, model)

    ###  计算train样本的softmax输出，附加了train样本的edge_index
    train_edge_index = edgeIndexOnCut(data.edge_index, 139)
    edge_weight = None
    conv1_out = F.relu(model.conv1(train_data, train_edge_index, edge_weight))
    conv2_in = conv1_out
    conv2_out = model.conv2(conv2_in, train_edge_index, edge_weight)
    logits2 = F.log_softmax(conv2_out, dim=1)
    pred2= logits2.max(1)[1]
    ###
    # for _, mask in data('train_mask', 'val_mask', 'test_mask'):   # 第一次返回参数是key，第二个是value。
    #     pred = logits[mask].max(1)[1]    # softmax中的最大值即为预测的类别。  mask是用来标识哪些节点属于训练集，哪些属于验证集，哪些属于测试集。
    #     acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()    # 把推断出来的训练(验证，测试)的类别值pred与训练(验证，测试)数据中的标签y进行比较后除以相应数据集的样本数得到准确率。
    #     accs.append(acc)
    # for mask in (data.y[0:140], data.y[140:640], data.y[1708:2708]):   # 第一次返回参数是key，第二个是value。
    #     pred = logits[mask].max(1)[1]    # softmax中的最大值即为预测的类别。  mask是用来标识哪些节点属于训练集，哪些属于验证集，哪些属于测试集。
    #     acc = pred.eq(mask).sum().item() / mask.sum().item()    # 把推断出来的训练(验证，测试)的类别值pred与训练(验证，测试)数据中的标签y进行比较后除以相应数据集的样本数得到准确率。
    #     accs.append(acc)
    mask = data.y[0:140]   # 第一次返回参数是key，第二个是value。
    pred = logits.max(1)[1]    # softmax中的最大值即为预测的类别。  mask是用来标识哪些节点属于训练集，哪些属于验证集，哪些属于测试集。
    pred1 = logtis1.max(1)[1]
    # acc = acc_single(pred, mask)
    # acc = pred.eq(mask).sum().item() / 140  # 把推断出来的训练(验证，测试)的类别值pred与训练(验证，测试)数据中的标签y进行比较后除以相应数据集的样本数得到准确率。
    # accs.append(acc)
    # return accs
    accs = acc_set(data_list, label_list, model)
    return accs

@torch.no_grad()
def test_onGraph(data_list, label_list, edge_index_list):
    model.eval()   # 不启用 BatchNormalization 和 Dropout，不计算梯度，因为这里只是在做预测
    accs = acc_set_onGraph(data_list, label_list, edge_index_list, model)
    return accs

def acquire_label(dataset):
    logits_supervisor = model_test(dataset, model)
    return logits_supervisor

def acquire_label_onGraph(dataset, edge_index):
    logits_supervisor_onGraph = model_test_onGraph(dataset, edge_index, model)
    return logits_supervisor_onGraph

# best_val_acc = test_acc = 0
# for epoch in range(1, 201):    # 样本的训练，记录每一个epoch的train，val和test
#     train()
#     train_acc, val_acc, tmp_test_acc = test()
#     if val_acc > best_val_acc:
#         best_val_acc = val_acc
#         test_acc = tmp_test_acc
#     log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
#     print(log.format(epoch, train_acc, best_val_acc, test_acc))

# def gcnRun():
#     best_val_acc = test_acc = 0
#     for epoch in range(1, 201):  # 样本的训练，记录每一个epoch的train，val和test
#         train()
#         train_acc, val_acc, tmp_test_acc = test()
#         if val_acc > best_val_acc:
#             best_val_acc = val_acc
#             test_acc = tmp_test_acc
#         log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
#         print(log.format(epoch, train_acc, best_val_acc, test_acc))
#     torch.save(train, 'net.plk')
#     net = torch.load('net.plk')

def gcnRun():
    best_val_acc = test_acc = 0
    train_data = data.x[0:140, :]
    val_data = data.x[140:640, :]
    test_data = data.x[1708:2708, :]
    data_list = [train_data, val_data, test_data]
    train_label = data.y[0:140]
    val_label = data.y[140:640]
    test_label = data.y[1708:2708]
    label_list = [train_label, val_label, test_label]

    train_edge_index = edgeIndexOnCut(data.edge_index, 139)
    val_edge_index = edgeIndexOnRange(data.edge_index, 140, 639) - 140   # 减去140是为了让坐标调整到数据集的内部，不至于下标超出数据集的大小。
    test_edge_index = edgeIndexOnRange(data.edge_index, 1708, 2707) - 1708 # 减去1708是为了让坐标调整到数据集的内部，不至于下标超出数据集的大小。
    edge_index_list = [train_edge_index, val_edge_index, test_edge_index]


    for epoch in range(1, 201):  # 样本的训练，记录每一个epoch的train，val和test
        train()
        # train_acc = test()
        # log = 'Epoch: {:03d}, Train: {:.4f}'
        # print(log.format(epoch, train_acc))
        # train_acc, val_acc, tmp_test_acc = test(data_list, label_list)  # 以用已经训练好的模型进行测试
        train_acc, val_acc, tmp_test_acc = test_onGraph(data_list, label_list, edge_index_list)  # 以用已经训练好的模型进行测试
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        print(log.format(epoch, train_acc, best_val_acc, test_acc))

    print('test_logits', acquire_label(test_data))
    torch.save(acquire_label(test_data), 'test_logits.plk')
    torch.save(train(), 'supervisor_train.plk')
    torch.save(test(data_list, label_list), 'supervisor_test.plk')
    torch.save(model(), 'model.plk')
    torch.save(acquire_label_onGraph(test_data, test_edge_index), 'test_logits_onGraph.plk')



gcnRun()  # 运行GCN，查看其在训练集、测试集上的准去率。
