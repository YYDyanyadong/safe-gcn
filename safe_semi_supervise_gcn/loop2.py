import torch

from torch_geometric.utils.num_nodes import maybe_num_nodes


def contains_self_loops2(edge_index):
    r"""Returns :obj:`True` if the graph given by :attr:`edge_index` contains
    self-loops.

    Args:
        edge_index (LongTensor): The edge indices.

    :rtype: bool
    """
    row, col = edge_index
    mask = row == col
    return mask.sum().item() > 0


def remove_self_loops2(edge_index, edge_attr=None):
    r"""Removes every self-loop in the graph given by :attr:`edge_index`, so
    that :math:`(i,i) \not\in \mathcal{E}` for every :math:`i \in \mathcal{V}`.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    row, col = edge_index
    mask = row != col
    edge_attr = edge_attr if edge_attr is None else edge_attr[mask]
    edge_index = edge_index[:, mask]

    return edge_index, edge_attr


def segregate_self_loops2(edge_index, edge_attr=None):
    r"""Segregates self-loops from the graph.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`, :class:`LongTensor`,
        :class:`Tensor`)
    """

    mask = edge_index[0] != edge_index[1]
    inv_mask = ~mask

    loop_edge_index = edge_index[:, inv_mask]
    loop_edge_attr = None if edge_attr is None else edge_attr[inv_mask]
    edge_index = edge_index[:, mask]
    edge_attr = None if edge_attr is None else edge_attr[mask]

    return edge_index, edge_attr, loop_edge_index, loop_edge_attr


def add_self_loops2(edge_index, edge_weight=None, fill_value=1, num_nodes=None):
    r"""Adds a self-loop :math:`(i,i) \in \mathcal{E}` to every node
    :math:`i \in \mathcal{V}` in the graph given by :attr:`edge_index`.
    In case the graph is weighted, self-loops will be added with edge weights
    denoted by :obj:`fill_value`.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_weight (Tensor, optional): One-dimensional edge weights.
            (default: :obj:`None`)
        fill_value (int, optional): If :obj:`edge_weight` is not :obj:`None`,
            will add self-loops with edge weights of :obj:`fill_value` to the
            graph. (default: :obj:`1`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    loop_index = torch.arange(0, num_nodes, dtype=torch.long,
                              device=edge_index.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)

    if edge_weight is not None:
        assert edge_weight.numel() == edge_index.size(1)
        loop_weight = edge_weight.new_full((num_nodes, ), fill_value)
        edge_weight = torch.cat([edge_weight, loop_weight], dim=0)

    edge_index = torch.cat([edge_index, loop_index], dim=1)

    return edge_index, edge_weight


def add_remaining_self_loops2(edge_index, edge_weight=None, fill_value=1,
                             num_nodes=None):
    r"""Adds remaining self-loop :math:`(i,i) \in \mathcal{E}` to every node
    :math:`i \in \mathcal{V}` in the graph given by :attr:`edge_index`.
    In case the graph is weighted and already contains a few self-loops, only
    non-existent self-loops will be added with edge weights denoted by
    :obj:`fill_value`.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_weight (Tensor, optional): One-dimensional edge weights.
            (default: :obj:`None`)
        fill_value (int, optional): If :obj:`edge_weight` is not :obj:`None`,
            will add self-loops with edge weights of :obj:`fill_value` to the
            graph. (default: :obj:`1`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    print('in the first num_nodes=', num_nodes)
    num_nodes = maybe_num_nodes(edge_index, num_nodes)  # 得到节点数2708
    print('after maybe_num_nodes:', num_nodes)
    row, col = edge_index   # 这里的row和col分别代表edge_index的第一和第二行，其长度都是10556
    print('in the loop the shape of edge_index is :', edge_index.shape)
    # for n, m in zip(row, col):   # 可以用zip的方式把row和col都遍历一遍。
    # print(n, m)

    mask = row != col     # mask的shape也是10556,和row与col的长度相同，说明没有一条边是环，于是所有的对角线值都应该加1

    if edge_weight is not None:
        assert edge_weight.numel() == edge_index.size(1)   # numel()用来返回元素个数
        inv_mask = ~mask   # ~mask是二进制取反的意思。 mask中的每一个值都是true。因此inv_mask中的每一个值都是false
        # print('row[inv_mask]:', row[inv_mask])
        loop_weight = torch.full(
            (num_nodes, ), fill_value,
            dtype=None if edge_weight is None else edge_weight.dtype,
            device=edge_index.device)   # torch.full()的作用是吧第二个参数(标量)，填满第一个参数形状的张量tensor，也就是[2708]的数值1
        remaining_edge_weight = edge_weight[inv_mask]   # 把所有的边的权重都赋值为false了, 所以这时候remaining_edge_weight是空的，于是remaining_edge_weight.numel()=0
        if remaining_edge_weight.numel() > 0:
            loop_weight[row[inv_mask]] = remaining_edge_weight   # 这里的row[inv_mask]就是空
        print('loop_weight:', loop_weight.shape)    # loop_weight是一个数值全是1的张量，其shape=[2708]
        # edge_weight = torch.cat([edge_weight[mask], loop_weight], dim=0)   # 最后计算得到的edge_weight的shape=[10556+2708=13264]，每一个位置的值都为1.
        edge_weight = torch.cat([edge_weight[0:637], loop_weight],
                                dim=0)  # 最后计算得到的edge_weight的shape=[10556+2708=13264]，每一个位置的值都为1.

    loop_index = torch.arange(0, num_nodes, dtype=row.dtype, device=row.device)   # 形成一个从0~num_nodes-1的tensor类型数组
    print('in the loop the shape of loop_index :', loop_index.shape)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)  # 在loop_index的第一个位置增加一个维度,并且把这个一维的数组变成了二维的。这里的repeat(2,1)把数据沿着第一个维度复制2遍，再延第二个维度复制1遍。
    print('in the loop the shape of loop_index:', loop_index.shape)
    print('in the loop the shape of loop_index the mask:', mask.shape)
    # edge_index = torch.cat([edge_index[:, mask], loop_index], dim=1) # 让edge_index[:,mask]与loop_index在第二个维度进行拼接,前者的shape=[2,10556]后者shape=[2,2708]，两者拼接以后就是[2,13264]
    edge_index = torch.cat([edge_index[0:637, 0:637], loop_index], dim=1)
    print('in the loop the shape of edge_index, loop_index:', edge_index.shape, loop_index.shape)

    return edge_index, edge_weight
