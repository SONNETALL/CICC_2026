import numpy as np


def quantize_uint4(x_fp):
    """
    把输入 x 从 [0,1] 浮点量化到 uint4，即 [0,15]
    返回:
        x_q: int32
        scale: 反量化比例
    """
    x_fp = np.clip(x_fp, 0.0, 1.0)
    scale = 15.0
    x_q = np.round(x_fp * scale).astype(np.int32)
    return x_q, scale


def quantize_int4_symmetric(w_fp):
    """
    对权重做对称 int4 量化到 [-8, 7]
    返回:
        w_q: int32
        scale: 浮点权重 ≈ w_q / scale
    """
    max_abs = np.max(np.abs(w_fp))
    if max_abs < 1e-8:
        scale = 1.0
    else:
        scale = 7.0 / max_abs

    w_q = np.round(w_fp * scale)
    w_q = np.clip(w_q, -8, 7).astype(np.int32)
    return w_q, scale


def dequant_matmul_output(y_int, x_scale, w_scale):
    """
    若 x_q ≈ x_fp * x_scale
       w_q ≈ w_fp * w_scale
    则 y_fp ≈ (x_q @ w_q) / (x_scale * w_scale)
    """
    return y_int.astype(np.float32) / (x_scale * w_scale)


def add_bias_fp32(y_fp, b_fp):
    return y_fp + b_fp.reshape(1, -1)
