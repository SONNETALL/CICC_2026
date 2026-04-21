# Copyright © [2025] Shanghai Xizhi Technology Co., Ltd. or its affiliates. All rights reserved.

import os
import numpy as np
import osimulator
import entrance
from sys import argv
from osimulator.api import load_gazelle_model

if __name__ == "__main__":
    # load model instance
    model = load_gazelle_model()

    b = 1
    m = 3136
    k = 756
    n = 64

    # b = 1000
    # m = 1
    # k = 8
    # n = 2
    in_bit = 4
    wght_bit = 4
    out_bit = 12
    input_type = "uint4"

    input_tensors = np.random.randint(low=0,
                                      high=2 ** in_bit - 1,
                                      size=(b, m, k),
                                      dtype=np.int32)
    wght_tensors = np.random.randint(low=-2 ** (wght_bit - 1),
                                     high=2 ** (wght_bit - 1),
                                     size=(b, k, n),
                                     dtype=np.int32)

    exp = np.matmul(input_tensors.astype(np.float32), wght_tensors.astype(np.float32))

    result_model = model(input_tensors, wght_tensors, inputType=input_type)
    entrance.dump_stats(result_model.numpy(), exp, in_bit, out_bit)