import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.points_cc = configs.points_cc
        self.sem_num = configs.sem_num
        self.bb_num = configs.ins_max_num

    # 定义模型的前向传播，即如何根据输入X返回所需的模型输出
    def forward(self, X):
        # 注意，这里我们使用ReLU的函数版本，其在nn.functional模块中定义。
        return self.out(F.relu(self.hidden(X)))