import torch
import os.path as osp
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

A=torch.tensor([1])
B=torch.tensor([4,5,6,4,4,5,6,7,8])
C=torch.cat((A,B))
print(C)
D = C[[0,2,3]]
print(D)
E=torch.tensor([[4], [5]])
bool_index = torch.eq(B, E)
print(bool_index)
# B[bool_index] = [[9], [9]]
print(bool_index[0:2])
print(len(E))
for i in range(0, len(E)):
    B[bool_index[i]] = E[i]
    print(B[bool_index[i]])
print(B)

def fun1(b):
    b[5] = 10

S = B
fun1(S)

print('B:', B)

a1 = torch.tensor([False, True, True, True])
a2 = torch.tensor([False, True])
a = torch.cat([a1, a2])
print(a)


dataset = 'Pubmed'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
data_Pubmed = dataset[0]

dataset = 'Citeseer'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
data_Citeseer = dataset[0]

