# train_mnist.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import os


# 定义极其简单的两层全连接网络
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(784, 128, bias=False)  # 对应模拟器的矩阵乘法，先不用bias
        self.fc2 = nn.Linear(128, 10, bias=False)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train():
    # 准备 MNIST 数据
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    model = SimpleMLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 简单训练 3 个 Epoch 即可轻松达到 90%+ 准确率
    model.train()
    for epoch in range(3):
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1} done.")

    # 提取 W1 和 W2 的权重并保存为 numpy 数组
    W1 = model.fc1.weight.data.numpy().T  # 注意转置，使形状变为 (784, 128)
    W2 = model.fc2.weight.data.numpy().T  # (128, 10)

    np.save('w1.npy', W1)
    np.save('w2.npy', W2)
    print("模型训练完成，权重已保存为 w1.npy 和 w2.npy")


if __name__ == "__main__":
    train()
