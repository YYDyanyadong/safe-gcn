import torch
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
from MyApplication.safe_semi_supervise_gcn.loop2 import add_remaining_self_loops2

from torch_geometric.nn.inits import glorot, zeros


class GCNConv2(MessagePassing):
    r"""The graph convolutional operator from the `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        normalize (bool, optional): Whether to add self-loops and apply
            symmetric normalization. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels, out_channels, improved=False, cached=False,
                 bias=True, normalize=True, **kwargs):
        super(GCNConv2, self).__init__(aggr='add', **kwargs)
        # 各参数赋值
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.normalize = normalize

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        self.cached_result = None
        self.cached_num_edges = None


    """
    set my owen parameter
    """
    # num_nodes = 140
    """
    set my owen parameter
    """
    '''
    在样本集中train的下标idx是0-139，共140个
    test的idx范围是140-639共500个样本，val的idx范围是1708-2707共1000个样本
    
    如果不算上环的话，那么edge总共有10556条，大小是[2,10556]，行标的第一个分量是边的起点，第二分量是相应的边的终点。
    边中节点的坐标与样本集中节点的坐标相对应，其中：
    0~637是下标0-139的train的样本集中节点对应的边的起点。
    638~2508是下标140-639的test的样本集中节点对应的边的起点。
    6844~10555是下标1708-2707的val的样本集中节点对应的起点。
    '''


    @staticmethod
    # norm的作用只在于完成了D^(-1/2)*A*D^(-1/2)形式的归一化，真正的卷积在forward中实现
    # edge_weight一开始是一个bool值,等于none
    def norm(edge_index, num_nodes, edge_weight=None, improved=False,
             dtype=None):   # 这里的edge_index的shape=[2,n]，n维边数*2，第一行存储边的起点，第二行存储边的终点。传入的shape=[2,10556],总共有5278条边
        print('the first sight in the conv the shape of edge_index', edge_index.shape)
        print('the first sight in the conv the edge_weight:', edge_weight)

        if edge_weight is None:   # cora数据集里的edge_weight就是没有值，所以如果有边就设置其为1
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)   # edge_index.size(1)记录的是edge_index的第二维的长度，也就是边数*2,这里的edeg_weight只是一个向量，其值为1代表该边不为空

        # 使得A=A+I或者A=A+2I取决于fill_value的值,更新后的A赋给edge_weight, 更新后的edge_index的shape=[2,13264], edge_weight的shape=[13264]，原大小是10556
        # 边是5278条，因为是无向边，所以边的条数×2，同时加上每个节点(2708个)自身的全以后=5278*2+2708=13264。
        # 需要注意的是所有的圈都加在了edge_index和edge_weight的尾部。
        fill_value = 1 if not improved else 2    # A=A+I或者A=A+2I，用来决定圈的度是算作1还是2
        edge_index, edge_weight = add_remaining_self_loops2(
            edge_index, edge_weight, fill_value, num_nodes)
        # 可以看出13264=10556(边数*2)+2708(节点数),edge_weight和edge_index都是在原始的edge_weight和edge_index上加上了对节点对自身的环的描述，也就是加上了矩阵A自身。
        print('in the conv the shape of edge_weight = ', edge_weight.shape)
        # row存储的是边的起点,col存储的是边的终点。
        row, col = edge_index
        print('the second sight in the conv the shape of edge_index', edge_index.shape)
        print('the second sight in the conv the shape of edge_weight:', edge_weight, edge_weight.shape)
        print('the second sight in the conv the num_nodes=', num_nodes)
        """特别注意一下scatter_add的作用，第一个参数是需要被计算的数值
        是一个全1的长度为13264的张量，第二个参数是下标位置，由于每个节点有环或者边，因此row存储的每个开始节点的下标会出现多次，比如[0,0,1,1,2,2,2,...]
        这个说明以0为起点的边有两条，1为起点的边有两条，2为起点的边有三条，dim=0表示沿着第0维相加,dim_size=num_nodes，这个num_nodes应该等于row中不重复出现的下标的个数
        最后的输出就是把edge_weight中对应于row位置的数据都相加起来，其值保存在对应下标中，在示例中最后的输出out=[2,2,3]
        再举一个例子，如果edge_weight=[1,1,3,2,5,1], row=[0,0,0,1,2,2],dim=0,dim_size=3，那么最后的输出out=[5,2,6]，对应下标是[0,1,2]"""
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)  # 对每一行的数据进行相加得到对角度矩阵D, edge_weight和row的shape=[13264], num_nodes=2708, deg的shape=[2708]
        print('row:', row, row.shape)
        print('col：', col, col.shape)
        print('deg:', deg, deg.shape)
        deg_inv_sqrt = deg.pow(-0.5)   # D^(-1/2)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0    # 对deg_inv_sqrt中为无穷的元素值赋为0
        print('the shape of edge_index, deg_inv_sqrt:', edge_weight.shape, deg_inv_sqrt.shape)
        # deg_inv_sqrt的shape=[2708]，row的shape=[13264]，不过row中不重复的下标只有2708个，从0~2707
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]   # 利用对应元素相乘的形式实现D^(-1/2)*A*D^(-1/2)
        # edge_index的shape=[2,13264], deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]的shape=[13264]

    def forward(self, x, edge_index, edge_weight=None):    # 完成卷积操作。注意，在这里节点的特征向量并没有出现。
        """"""
        x = torch.matmul(x, self.weight)   # 数据x的特征向量与权重矩阵weight相乘，这个weight是我们需要学习的参数,weight可以看作是特征维数的一种映射，可以降维也可以升维。

        if self.cached and self.cached_result is not None:    # 如果在每一层的卷积中重复使用D^(-1/2)*A*D^(-1/2)
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}. Please '
                    'disable the caching behavior of this layer by removing '
                    'the `cached=True` argument in its constructor.'.format(
                        self.cached_num_edges, edge_index.size(1)))

        if not self.cached or self.cached_result is None:   # 不重复使用D^(-1/2)*A*D^(-1/2)，在每一层都更新A和D
            self.cached_num_edges = edge_index.size(1)   # 数据中节点数
            if self.normalize:
                edge_index, norm = self.norm(edge_index, x.size(self.node_dim),
                                             edge_weight, self.improved,
                                             x.dtype)   # 返回的norm是重新计算的D^(-1/2)*A*D^(-1/2)
            else:
                norm = edge_weight   # 如果不进行标准化的话norm就是原始的edge_weight
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result

        return self.propagate(edge_index, x=x, norm=norm)   # 完成D^(-1/2)*A*D^(-1/2)*H(i)*W(i)=H(i+1), 通过不断调用forward那么D^(-1/2)*A*D^(-1/2)*H(i+1)*W(i+1)=H(i+2)，其中A和D的值可以用第一次计算的值也可以用不断更新后的值。

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j if norm is not None else x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
