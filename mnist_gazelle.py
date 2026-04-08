import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import time
from osimulator.api import load_gazelle_model

# ===== 1. 加载光计算模型 =====
model = load_gazelle_model()

# ===== 2. 加载MNIST =====
transform = transforms.ToTensor()
testset = torchvision.datasets.MNIST(
    root='/workspace/data',
    train=False,
    download=True,
    transform=transform
)

# ===== 3. 定义模型 =====
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(28*28, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        return self.fc(x)

net = Net()
net.load_state_dict(torch.load("/workspace/mnist.pth"))
net.eval()

# ===== 4. 提取权重 =====
weight = net.fc.weight.detach().numpy().T
weight = weight / np.max(np.abs(weight)) * 7
weight = weight.astype(np.int32)

# ===== 5. 推理 =====
correct = 0
total = 100

gazelle_time = 0
cpu_time = 0

for i in range(total):
    image, label = testset[i]

    # ===== 输入处理 =====
    input_data = image.view(1, -1).numpy()
    input_data_q = (input_data * 15).astype(np.int32)

    input_tensors = input_data_q.reshape(1, 1, 784)
    wght_tensors = weight.reshape(1, 784, 10)

    # ===== 光计算（Gazelle） =====
    start = time.time()
    result = model(input_tensors, wght_tensors, inputType="uint4")
    end = time.time()

    gazelle_time += (end - start)

    output = result.numpy().reshape(-1)
    pred = np.argmax(output)

    if pred == label:
       correct += 1

    # ===== CPU 推理 =====
    with torch.no_grad():
        start = time.time()
        cpu_out = net(image.unsqueeze(0))
        end = time.time()

        cpu_time += (end - start)

# ===== 结果 =====
accuracy = correct / total

gazelle_latency = (gazelle_time / total) * 1000
cpu_latency = (cpu_time / total) * 1000

# 光计算占比（用时间占比表示）
ratio = gazelle_time / (gazelle_time + cpu_time)

print("Top1 Accuracy:", accuracy)
print("Gazelle Latency: %.3f ms" % gazelle_latency)
print("CPU Latency: %.3f ms" % cpu_latency)
print("Optical Computing Ratio: %.2f%%" % (ratio * 100))
