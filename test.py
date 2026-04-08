# -*- coding: utf-8 -*-

    # import numpy as np
    # import torch
    # import torch.nn as nn
    # import time
    # from PIL import Image
    # from osimulator.api import load_gazelle_model
    #
    # # ===== 1. 加载光计算模型 =====
    # model = load_gazelle_model()
    #
    # # ===== 2. 定义模型结构 =====
    # class Net(nn.Module):
    #     def __init__(self):
    #         super().__init__()
    #         self.fc = nn.Linear(28*28, 10)
    #
    #     def forward(self, x):
    #         x = x.view(-1, 28*28)
    #         return self.fc(x)
    #
    # net = Net()
    # net.load_state_dict(torch.load("/workspace/mnist.pth"))
    # net.eval()
    #
    # # ===== 3. 加载图片 =====
    # img_path = "/workspace/test.png"
    #
    # img = Image.open(img_path).convert('L')
    # img = img.resize((28, 28))
    #
    # img_np = np.array(img)
    #
    # # 归一化（0~1）
    # img_np = img_np / 255.0
    #
    # # ===== 自动判断是否反色 =====
    # mean_val = np.mean(img_np)
    # print("图像平均亮度:", mean_val)
    #
    # if mean_val > 0.5:
    #     print("检测到白底黑字 → 执行反色")
    #     img_np = 1 - img_np
    # else:
    #     print("检测到黑底白字 → 不需要反色")
    #
    # # ===== 打印输入信息（调试）=====
    # print("\n===== 输入调试 =====")
    # print("原始输入（0~1）:")
    # print(img_np.reshape(28,28))
    #
    #
    # # ===== 4. 准备输入 =====
    # input_data = img_np.reshape(1, -1)
    #
    # # 🔥 量化（防溢出）
    # input_data_q = np.clip(input_data * 15, 0, 15).astype(np.int32)
    #
    # input_tensors = input_data_q.reshape(1, 1, 784)
    # print("\n量化后输入（0~15）:")
    # print(input_data_q.reshape(28,28))
    #
    # # ===== 5. 权重 =====
    # weight = net.fc.weight.detach().numpy().T
    # weight = weight / np.max(np.abs(weight)) * 7
    # weight = weight.astype(np.int32)
    #
    # wght_tensors = weight.reshape(1, 784, 10)
    #
    # # ===== 6. 光计算推理 =====
    # start = time.time()
    # result = model(input_tensors, wght_tensors, inputType="uint4")
    # end = time.time()
    #
    # gazelle_latency = (end - start) * 1000
    #
    # output = result.numpy().reshape(-1)
    # pred = np.argmax(output)
    #
    # # ===== 7. CPU推理 =====
    # with torch.no_grad():
    #     input_tensor = torch.tensor(input_data, dtype=torch.float32)
    #
    #     start = time.time()
    #     cpu_out = net(input_tensor)
    #     end = time.time()
    #
    #     cpu_latency = (end - start) * 1000
    #
    # # ===== 8. 光计算占比 =====
    # ratio = gazelle_latency / (gazelle_latency + cpu_latency)
    #
    # # ===== 9. 输出结果 =====
    # print("\n===== 推理结果 =====")
    # print("预测结果:", pred)
    # print("Gazelle Latency: %.3f ms" % gazelle_latency)
    # print("CPU Latency: %.3f ms" % cpu_latency)
    # print("Optical Computing Ratio: %.2f%%" % (ratio * 100))
    #
    # # ===== 10. 可视化输入（可选）=====
    # try:
    #     import matplotlib.pyplot as plt
    #     plt.imshow(img_np.reshape(28,28), cmap='gray')
    #     plt.title("Input Image (after preprocess)")
    #     plt.show()
    # except:
    #     print("未安装matplotlib，跳过可视化")
import numpy as np
import torch
import torch.nn as nn
import time
from PIL import Image
from osimulator.api import load_gazelle_model

# ===== 1. 加载光计算模型 =====
model = load_gazelle_model()

# ===== 2. 定义模型 =====
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

# ===== 3. 加载图片 =====
img_path = "/workspace/test.png"
img = Image.open(img_path).convert('L')

# ===== 转numpy =====
img_np = np.array(img)

# ===== 自动反色判断 =====
img_norm = img_np / 255.0
mean_val = np.mean(img_norm)
print("图像平均亮度:", mean_val)

if mean_val > 0.5:
    print("检测到白底黑字 → 执行反色")
    img_np = 255 - img_np
else:
    print("检测到黑底白字 → 不需要反色")

# ===== 二值化（提取数字区域）=====
img_bin = (img_np > 50).astype(np.uint8)

coords = np.argwhere(img_bin)

if len(coords) > 0:
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    img_crop = img_np[y_min:y_max+1, x_min:x_max+1]
else:
    img_crop = img_np

# ===== resize =====
img = Image.fromarray(img_crop)
img = img.resize((20, 20))   # 先缩小

# ===== 居中到28x28 =====
canvas = np.zeros((28, 28))
img_small = np.array(img)

h, w = img_small.shape
y_offset = (28 - h) // 2
x_offset = (28 - w) // 2

canvas[y_offset:y_offset+h, x_offset:x_offset+w] = img_small

# ===== 归一化 =====
img_np = canvas / 255.0

# ===== 打印输入 =====
np.set_printoptions(threshold=np.inf)

print("\n===== 输入调试 =====")
print("原始输入（28x28）:")
print(img_np)

# ===== 准备输入 =====
input_data = img_np.reshape(1, -1)

# ===== 量化 =====
input_data_q = np.clip(input_data * 15, 0, 15).astype(np.int32)

print("\n量化后输入（28x28）:")
print(input_data_q.reshape(28,28))

input_tensors = input_data_q.reshape(1, 1, 784)

# ===== 权重 =====
weight = net.fc.weight.detach().numpy().T
weight = weight / np.max(np.abs(weight)) * 7
weight = weight.astype(np.int32)

wght_tensors = weight.reshape(1, 784, 10)

# ===== 光计算 =====
start = time.time()
result = model(input_tensors, wght_tensors, inputType="uint4")
end = time.time()

gazelle_latency = (end - start) * 1000

output = result.numpy().reshape(-1)
pred = np.argmax(output)

# ===== CPU推理 =====
with torch.no_grad():
    input_tensor = torch.tensor(input_data, dtype=torch.float32)

    start = time.time()
    cpu_out = net(input_tensor)
    end = time.time()

    cpu_latency = (end - start) * 1000

# ===== 光计算占比 =====
ratio = gazelle_latency / (gazelle_latency + cpu_latency)

# ===== 输出 =====
print("\n===== 推理结果 =====")
print("预测结果:", pred)
print("Gazelle Latency: %.3f ms" % gazelle_latency)
print("CPU Latency: %.3f ms" % cpu_latency)
print("Optical Computing Ratio: %.2f%%" % (ratio * 100))

# ===== 可视化（可选）=====
try:
    import matplotlib.pyplot as plt
    plt.imshow(img_np, cmap='gray')
    plt.title("Processed Input")
    plt.show()
except:
    pass
