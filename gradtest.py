import torch
import torch.nn.functional as F
from torch import nn


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(5, 4)
        self.fc_mu = nn.Linear(4, 1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = self.fc_mu(x)
        return x

if __name__ == "__main__":
    # x = torch.randn(2, 5)
    # a = Actor(None)
    # x.requires_grad = True
    # y = torch.randn(2, 1)
    # optimizer = torch.optim.Adam(list(a.parameters()), lr=0.001)
    # optimizer.zero_grad()
    # loss = F.mse_loss(a(x), y)
    # print(type(a.parameters()))
    # f1 = torch.autograd.grad(loss, a.parameters(), create_graph=True)[0]
    #
    # print(f1)
    # print("1", x.grad)
    # print("22",a.fc1.weight.grad)
    # loss2  = F.mse_loss(x.grad,torch.ones_like(x.grad))
    # loss2.backward()
    # print("3", x.grad)


    # 创建一个需要计算导数的张量
    x = torch.tensor(1.321, requires_grad=True)
    z = torch.tensor(2.31, requires_grad=True)
    # 计算一阶导数
    y = x ** 3.21 + (z**1.3)* (x**2)* 2.99
    loss = F.mse_loss(y,torch.ones_like(y))
    loss.backward( create_graph=True)
    print(x.grad)
    x.grad.backward()
    print(x.grad)
    # print("@@", torch.autograd.grad([y], [x,z], create_graph=True))
    # dy_dx = torch.autograd.grad(y, x, create_graph=True)[0]
    # print("@@",x.grad)
    # print("一阶导数 dy/dx =", dy_dx)
    # # 计算二阶导数
    # Ydxdz= torch.autograd.grad(dy_dx, z)
    #
    #
    # print("二阶导数 d2y/dx2 =", Ydxdz)
