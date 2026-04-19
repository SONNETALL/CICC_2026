# # train_mnist.py
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import datasets, transforms
# import numpy as np
# import os


# # 定义极其简单的两层全连接网络
# class SimpleMLP(nn.Module):
#     def __init__(self):
#         super(SimpleMLP, self).__init__()
#         self.fc1 = nn.Linear(784, 128, bias=False)  # 对应模拟器的矩阵乘法，先不用bias
#         self.fc2 = nn.Linear(128, 10, bias=False)

#     def forward(self, x):
#         x = x.view(-1, 784)
#         x = torch.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x


# def train():
#     # 准备 MNIST 数据
#     transform = transforms.Compose([transforms.ToTensor()])
#     train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
#     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

#     model = SimpleMLP()
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)

#     # 简单训练 3 个 Epoch 即可轻松达到 90%+ 准确率
#     model.train()
#     for epoch in range(3):
#         for data, target in train_loader:
#             optimizer.zero_grad()
#             output = model(data)
#             loss = criterion(output, target)
#             loss.backward()
#             optimizer.step()
#         print(f"Epoch {epoch + 1} done.")

#     # 提取 W1 和 W2 的权重并保存为 numpy 数组
#     W1 = model.fc1.weight.data.numpy().T  # 注意转置，使形状变为 (784, 128)
#     W2 = model.fc2.weight.data.numpy().T  # (128, 10)

#     np.save('w1.npy', W1)
#     np.save('w2.npy', W2)
#     print("模型训练完成，权重已保存为 w1.npy 和 w2.npy")


# if __name__ == "__main__":
#     train()

# train_mnist_noisy.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

# 定义带有噪声注入机制的全连接网络
class SimpleMLP_Noisy(nn.Module):
    def __init__(self, noise_level=0.05):
        super(SimpleMLP_Noisy, self).__init__()
        self.fc1 = nn.Linear(784, 128, bias=False)
        self.fc2 = nn.Linear(128, 10, bias=False)
        self.noise_level = noise_level
        print(f"模型已初始化，训练时将注入强度为 {self.noise_level} 的噪声。")

    def forward(self, x):
        x = x.view(-1, 784)
        
        # --- 噪声注入 ---
        # 仅在训练模式下注入权重噪声，模拟光芯片的权重加载误差
        if self.training:
            # 使用带噪声的权重进行前向传播
            noisy_w1 = self.fc1.weight + torch.randn_like(self.fc1.weight) * self.noise_level
            x = nn.functional.linear(x, noisy_w1) 
        else:
            # 在评估/推理模式下，使用原始权重
            x = self.fc1(x)
            
        x = torch.relu(x)
        
        # 仅在训练模式下注入激活噪声，模拟计算过程和探测器噪声
        if self.training:
            x = x + torch.randn_like(x) * self.noise_level
        
        x = self.fc2(x)
        return x


def train():
    # 准备 MNIST 数据
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    # 实例化带噪声的模型
    model = SimpleMLP_Noisy(noise_level=0.05)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练 3 个 Epoch
    model.train() # 确保模型处于训练模式，以激活噪声注入
    for epoch in range(3):
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        print(f"噪声感知训练 Epoch {epoch + 1} done.")

    # 切换到评估模式，将不再有噪声
    model.eval()
    
    # 提取 W1 和 W2 的权重并保存为 numpy 数组
    W1 = model.fc1.weight.data.numpy().T  # 转置以匹配 (input_dim, output_dim)
    W2 = model.fc2.weight.data.numpy().T

    np.save('w1.npy', W1)
    np.save('w2.npy', W2)
    print("模型训练完成，权重已保存为 w1.npy 和 w2.npy")


if __name__ == "__main__":
    train()