import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

import osimulator
import entrance
from osimulator.api import load_gazelle_model

from utils_quant import (
    quantize_uint4,
    quantize_int4_symmetric,
    dequant_matmul_output,
    add_bias_fp32,
)


class MLP(nn.Module):
    def __init__(self, in_dim=784, hidden_dim=256, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def load_test_csv(csv_path):
    df = pd.read_csv(csv_path)
    data = df.values
    y = data[:, 0].astype(np.int64)
    x = data[:, 1:].astype(np.float32) / 255.0
    return x, y


def simulator_linear(model, x_q, w_q, input_type="uint4"):
    """
    按官方示例接口调用 Gazelle：
    input_tensors: (b, m, k)
    wght_tensors : (b, k, n)

    这里我们令:
    b = 1
    m = batch_size
    k = in_dim
    n = out_dim
    """
    x_q_3d = np.expand_dims(x_q, axis=0).astype(np.int32)   # (1, m, k)
    w_q_3d = np.expand_dims(w_q, axis=0).astype(np.int32)   # (1, k, n)

    out = model(x_q_3d, w_q_3d, inputType=input_type)

    # 按示例 result_model.numpy() 推断可以转 numpy
    out_np = out.numpy()

    # 去掉 batch 维，得到 (m, n)
    out_np = out_np[0]
    return out_np


def relu(x):
    return np.maximum(x, 0.0)


def evaluate_ltsim(test_x, test_y, fp32_ckpt, batch_size=256):
    # 1. 读取训练好的浮点模型参数
    net = MLP(in_dim=784, hidden_dim=256, num_classes=10)
    state = torch.load(fp32_ckpt, map_location="cpu")
    net.load_state_dict(state)
    net.eval()

    fc1_w = net.fc1.weight.detach().cpu().numpy().T   # PyTorch是(out,in)，这里转成(in,out)
    fc1_b = net.fc1.bias.detach().cpu().numpy()

    fc2_w = net.fc2.weight.detach().cpu().numpy().T
    fc2_b = net.fc2.bias.detach().cpu().numpy()

    # 2. 权重量化一次即可
    fc1_w_q, fc1_w_scale = quantize_int4_symmetric(fc1_w)
    fc2_w_q, fc2_w_scale = quantize_int4_symmetric(fc2_w)

    # 3. 加载 Gazelle 模型
    sim_model = load_gazelle_model()

    total = 0
    correct = 0

    num_samples = test_x.shape[0]
    num_batches = (num_samples + batch_size - 1) // batch_size

    for bi in range(num_batches):
        st = bi * batch_size
        ed = min((bi + 1) * batch_size, num_samples)

        x_fp = test_x[st:ed]       # (m, 784)
        y_gt = test_y[st:ed]

        # ---- 第一层：输入量化 -> LT-Simulator matmul -> 反量化 -> 加bias -> ReLU ----
        x1_q, x1_scale = quantize_uint4(x_fp)   # (m, 784), uint4
        y1_int = simulator_linear(sim_model, x1_q, fc1_w_q, input_type="uint4")
        y1_fp = dequant_matmul_output(y1_int, x1_scale, fc1_w_scale)
        y1_fp = add_bias_fp32(y1_fp, fc1_b)
        y1_fp = relu(y1_fp)

        # ReLU 后再量化给第二层
        # 这里简单按 [0,1] 截断不太好，因此改为动态归一化到 uint4
        max_val = np.max(y1_fp)
        if max_val < 1e-8:
            y1_scale2 = 1.0
            y1_q = np.zeros_like(y1_fp, dtype=np.int32)
        else:
            y1_scale2 = 15.0 / max_val
            y1_q = np.round(y1_fp * y1_scale2)
            y1_q = np.clip(y1_q, 0, 15).astype(np.int32)

        # ---- 第二层：LT-Simulator matmul -> 反量化 -> 加bias ----
        y2_int = simulator_linear(sim_model, y1_q, fc2_w_q, input_type="uint4")
        y2_fp = y2_int.astype(np.float32) / (y1_scale2 * fc2_w_scale)
        y2_fp = add_bias_fp32(y2_fp, fc2_b)

        pred = np.argmax(y2_fp, axis=1)
        correct += np.sum(pred == y_gt)
        total += len(y_gt)

        if (bi + 1) % 10 == 0 or (bi + 1) == num_batches:
            print(f"[{bi+1}/{num_batches}] partial_acc={correct/total:.4f}")

    acc = correct / total
    print(f"Final LT-Simulator Top-1 Accuracy = {acc:.4f}")


def main():
    test_csv = "./data/mnist_test.csv"
    fp32_ckpt = "./ckpt/mnist_mlp_fp32.pth"

    test_x, test_y = load_test_csv(test_csv)
    evaluate_ltsim(test_x, test_y, fp32_ckpt, batch_size=256)


if __name__ == "__main__":
    main()
