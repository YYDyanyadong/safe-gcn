import torch
import numpy as np
from collections import Counter

train_mask_U = torch.tensor([False, True, True, False])
a = torch.tensor([1,3,4,5,6])
b = torch.tensor([8,8,8])
a[train_mask_U] = b
print(a)





