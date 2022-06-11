import torch  # 如果多次导入某个包，那么Pathon会默认第一次的导入场景。
import MyApplication.safe_semi_supervise_gcn as gcn2
import MyApplication.safe_semi_supervise_gcn.gcn_conv2 as gcn_op


# label1idx = torch.tensor(1)
# label1idx = label1idx.bool()  # 类型转换
# print(label1idx.type())
# print(label1idx.item())       # 取值
# print(gcn.data.train_mask.type())     # 查看数据类型
# label1idx_cuda = label1idx.cuda()
# label1idxbool = torch.eq(gcn.data.train_mask, label1idx_cuda)
# print(gcn.data.test_mask.size())
# print(label1idxbool)
# print((label1idxbool == label1idx_cuda).nonzero())  # torch中的eq函数是找到tensor中的指定数据并返回指定位置的布尔值，而equal返回的是一个布尔值。

# getDataIdx是找到每个数据集中被用到的数据的下标。因为1代表该数据被用到，0代码没有用到，所以要用到nonzero()这个函数

def getDataIdx(cora_data):
    # 定义一个tensor标量，并将其转换为true，且将数据放入到显卡的cuda中
    label_true = torch.tensor(1)
    label_true = label_true.bool()  # 类型转换
    label_true_cuda = label_true.cuda()

    data = cora_data
    # 通过torch.eq找到train数据集中所有为true的样本，并同样用true和fasle来表示
    # 通过nonzero找到train中为true的样本的下标index
    # torch中的eq函数是找到tensor中的指定数据并返回指定位置的布尔值，而equal返回的是一个布尔值。
    # data_bool = torch.eq(data, label_true_cuda)
    label_true_idx = (data == label_true_cuda).nonzero()
    return label_true_idx


# 样本集总共有2708个样本
# train的idx范围是0-139共140个，test的idx范围是140-639共500个样本，val的idx范围是1708-2707共1000个样本

# train_mask_idx = getDataIdx(gcn.data.train_mask)
# test_mask_idx = getDataIdx(gcn.data.test_mask)
# val_mask_idx = getDataIdx(gcn.data.val_mask)
#
# print(train_mask_idx, test_mask_idx, val_mask_idx)

def myTrain( data_mask ):
    data_mask = data_mask
    gcn2.model.train()
    gcn2.optimizer.zero_grad()
    gcn2.F.nll_loss(gcn2.model()[data_mask], gcn2.data.y[data_mask]).backward()

    gcn2.optimizer.step()

def myGcnRun():
    best_val_acc = test_acc = 0
    for epoch in range(1, 201):  # 样本的训练，记录每一个epoch的train，val和test
        myTrain(gcn2.data.train_mask)
        train_acc, val_acc, tmp_test_acc = gcn2.test()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        print(log.format(epoch, train_acc, best_val_acc, test_acc))


# myGcnRun()  # 运行我自己的代码
def datasetIndexEndOnEdge(edge_index, dataset_maxindex, choose):

    dataset_index_max_onEdge = torch.eq(edge_index[0, :], dataset_maxindex).nonzero()
    if choose == "max":
        dataset_index_end_onEdge = torch.max(dataset_index_max_onEdge, 0)
        return  dataset_index_end_onEdge[0].item()
    else:
        dataset_index_start_onEdge = torch.min(dataset_index_max_onEdge, 0)
        return dataset_index_start_onEdge[0].item()

# 获取train,test和val数据集在edge上的下标范围。
train_index_end_onEdge = datasetIndexEndOnEdge(gcn2.data.edge_index, 139, "max")
test_index_start_onEdge = datasetIndexEndOnEdge(gcn2.data.edge_index, 140, "min")
test_index_end_onEdge = datasetIndexEndOnEdge(gcn2.data.edge_index, 639, "max")
val_index_start_onEdge = datasetIndexEndOnEdge(gcn2.data.edge_index, 1708, "min")
val_index_end_onEdge = datasetIndexEndOnEdge(gcn2.data.edge_index, 2707, "max")

print(train_index_end_onEdge, test_index_start_onEdge, test_index_end_onEdge, val_index_start_onEdge, val_index_end_onEdge)


# 安全半监督的GCN
# def safe_semi_GCN():


# 构建一个由完全train所构成的有监督的GCN训练模型
# 1.这个模型的相似性矩阵只用到了train中的的数据，因此相似性矩阵一定要改。
# 2.先前的半监督GCN是用所有的样本构成了相似性矩阵，只是在计算损失函数的时候只用到了train_mask
# 3.因此新的代码中一开始的相似矩阵可能只能由很少一部分的L数据来构建。同时，半监督的GCN的相似性矩阵也要改，因为并非用到了所有的数据，因为至少有一部分数据是测试数据，这部分数据是不能用作建立半监督GCN的相似性矩阵的。

# 监督的GCN
def GCN_supervisor():
    # 构建只有train的相似性矩阵,其中矩阵A是要获得的相似性矩阵，是一个对称的方阵
    # 输入输出的edge_index的差别在于输出的edge_index加上了自身到自身的圈的权重
    edge_index, DaDunify = gcn_op.GCNConv2.norm(gcn2.data.edge_index[0:train_index_end_onEdge, :], 140, None, False, None)
    print(edge_index, DaDunify)
    #
    # def norm(edge_index, num_nodes, edge_weight=None, improved=False,
    #          dtype=None):   # 这里的edge_index的shape=[2,n]，n维边数*2，第一行存储边的起点，第二行存储边的终点。传入的shape=[2,10556],总共有5278条边
    #     print('the first sight in the conv the shape of edge_index', edge_index.shape)
    #
    #     if edge_weight is None:   # cora数据集里的edge_weight就是没有值，所以如果有边就设置其为1
    #         edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
    #                                  device=edge_index.device)   # edge_index.size(1)记录的是edge_index的第二维的长度，也就是边数*2,这里的edeg_weight只是一个向量，其值为1代表该边不为空
    #
    #     # 使得A=A+I或者A=A+2I取决于fill_value的值,更新后的A赋给edge_weight, 更新后的edge_index的shape=[2,13264], edge_weight的shape=[13264]，原大小是10556
    #     # 边是5278条，因为是无向边，所以边的条数×2，同时加上每个节点(2708个)自身的全以后=5278*2+2708=13264。
    #     # 需要注意的是所有的圈都加在了edge_index和edge_weight的尾部。
    #     fill_value = 1 if not improved else 2    # A=A+I或者A=A+2I，用来决定圈的度是算作1还是2
    #     edge_index, edge_weight = add_remaining_self_loops(
    #         edge_index, edge_weight, fill_value, num_nodes)
    #     # 可以看出13264=10556(边数*2)+2708(节点数),edge_weight和edge_index都是在原始的edge_weight和edge_index上加上了对节点对自身的环的描述，也就是加上了矩阵A自身。
    #     print('in the conv the shape of edge_weight = ', edge_weight.shape)
    #     # row存储的是边的起点,col存储的是边的终点。
    #     row, col = edge_index
    #     print('the second sight in the conv the shape of edge_index', edge_index.shape)
    #     print('the second sight in the conv the shape of edge_weight:', edge_weight)
    #     print('the second sight in the conv the num_nodes=', num_nodes)
    #     """特别注意一下scatter_add的作用，第一个参数是需要被计算的数值
    #     是一个全1的长度为13264的张量，第二个参数是下标位置，由于每个节点有环或者边，因此row存储的每个开始节点的下标会出现多次，比如[0,0,1,1,2,2,2,...]
    #     这个说明以0为起点的边有两条，1为起点的边有两条，2为起点的边有三条，dim=0表示沿着第0维相加,dim_size=num_nodes，这个num_nodes应该等于row中不重复出现的下标的个数
    #     最后的输出就是把edge_weight中对应于row位置的数据都相加起来，其值保存在对应下标中，在示例中最后的输出out=[2,2,3]
    #     再举一个例子，如果edge_weight=[1,1,3,2,5,1], row=[0,0,0,1,2,2],dim=0,dim_size=3，那么最后的输出out=[5,2,6]，对应下标是[0,1,2]"""
    #     deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)  # 对每一行的数据进行相加得到对角度矩阵D, edge_weight和row的shape=[13264], num_nodes=2708, deg的shape=[2708]
    #     print('row:', row)
    #     print('col：', col)
    #     print('deg:', deg.shape)
    #     deg_inv_sqrt = deg.pow(-0.5)   # D^(-1/2)
    #     deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0    # 对deg_inv_sqrt中为无穷的元素值赋为0
    #     print('the shape of edge_index, deg_inv_sqrt:', edge_weight.shape, deg_inv_sqrt.shape)
    #     # deg_inv_sqrt的shape=[2708]，row的shape=[13264]，不过row中不重复的下标只有2708个，从0~2707
    #     return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]   # 利用对应元素相乘的形式实现D^(-1/2)*A*D^(-1/2)
    #     # edge_index的shape=[2,13264], deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]的shape=[13264]

GCN_supervisor()