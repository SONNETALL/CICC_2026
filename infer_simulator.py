# infer_simulator.py
import numpy as np
from torchvision import datasets, transforms
import torch
from osimulator.api import load_gazelle_model


# 1. 简单的伪量化函数 (适配模拟器要求的位宽)
def quantize_tensor(tensor, bit=4, is_unsigned=False):
    # 此处为简化版的 Min-Max 量化，比赛中建议使用更严谨的量化方法
    max_val = np.max(np.abs(tensor))
    if max_val == 0:
        return np.zeros_like(tensor, dtype=np.int32)

    if is_unsigned:
        scale = (2 ** bit - 1) / max_val
        q_tensor = np.round(tensor * scale)
        q_tensor = np.clip(q_tensor, 0, 2 ** bit - 1)
    else:
        scale = (2 ** (bit - 1) - 1) / max_val
        q_tensor = np.round(tensor * scale)
        q_tensor = np.clip(q_tensor, -2 ** (bit - 1), 2 ** (bit - 1) - 1)

    return q_tensor.astype(np.int32)


def main():
    # 2. 加载光计算模拟器
    print("Loading Gazelle model...")
    gazelle_sim = load_gazelle_model()

    # 3. 加载 MNIST 测试数据
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(root='/workspace/data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)  # Batch=1 方便逐个推理

    # 4. 加载训练好的权重
    W1 = np.load('/workspace/w1.npy')  # Shape: (784, 128)
    W2 = np.load('/workspace/w2.npy')  # Shape: (128, 10)

    # 量化权重 (适配模拟器的 wght_bit=4)
    W1_quant = quantize_tensor(W1, bit=4, is_unsigned=False)

    # 扩展 W1 维度以匹配模拟器 API: (b, k, n) -> (1, 784, 128)
    W1_sim = np.expand_dims(W1_quant, axis=0)

    correct = 0
    total = 0

    print("开始使用光计算模拟器进行推理评估...")

    # 5. 推理循环
    for images, labels in test_loader:
        # 展平图像：(1, 1, 28, 28) -> (1, 784)
        X = images.view(-1, 784).numpy()

        # 量化输入 (适配模拟器的 in_bit=4, input_type="uint4")
        X_quant = quantize_tensor(X, bit=4, is_unsigned=True)

        # 扩展 X 维度以匹配模拟器 API: (b, m, k) -> (1, 1, 784)
        X_sim = np.expand_dims(X_quant, axis=0)

        # ---------------------------------------------------------
        # 【核心：使用光计算模拟器执行第一层矩阵乘法】
        # 输入 X_sim: (1, 1, 784)
        # 权重 W1_sim: (1, 784, 128)
        # 输出 out1_sim 将是 (1, 1, 128)
        # ---------------------------------------------------------
        result_model = gazelle_sim(X_sim, W1_sim, inputType="uint4")

        # 将模拟器结果转换为 numpy 并去除多余维度
        out1 = result_model.numpy().reshape(1, 128)

        # 反量化（由于是演示，这里直接将其当做特征继续传播）
        # ReLU 激活
        out1 = np.maximum(0, out1)

        # 第二层使用传统 NumPy 计算即可 (计算量很小，128x10)
        logits = np.matmul(out1, W2)

        # 预测结果
        pred = np.argmax(logits, axis=1)
        if pred[0] == labels.numpy()[0]:
            correct += 1
        total += 1

        if total % 1000 == 0:
            print(f"已处理 {total}/10000 样本，当前准确率: {100 * correct / total:.2f}%")

    print(f"\n评估完成！使用光计算模拟器的最终准确率: {100 * correct / total:.2f}%")


if __name__ == "__main__":
    main()
