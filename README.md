```text
├── data/                                 # 数据存储目录 (tips: 此文件夹在 .gitignore 中配置)
│   ├── raw/                              # 原始数据
│   └── processed/                        # 处理后的数据
├── docs/                                 # 项目文档，存放比赛资料、算法推导及硬件方案文档
├── models/                               # 模型权重存放区 (注: 默认由 .gitignore 忽略)
│   └── *.ckpt/*.pth                      # 存放训练好的权重文件
├── src/                                  # 核心源代码
│   └── train.py
├── tests/                                # 测试代码
│   └── test_gemm.py
├── configs/                              # 配置文件
│   └── config.yaml                       # YAML/JSON 格式参数配置
├── README.md                             # 项目简介与使用指南
└── .gitignore                            # Git 忽略规则配置

最终训练代码为./src/train_mnist_mlp_80_20.py   运行时加入--qat 以启用量化感知训练，在训练时使用uint8.此时默认fc1隐藏层的clip截断为225。  
最终推理测试代码为./test/test_mnist_mlp_80_20_optical.py 默认将三层权重都卸载到光计算，使用命令行参数可以进行调节。如python test_mnist_mlp_80_20_optical.py --optical-layers fc1//此为只卸载fc1层进入光计算的指令，此时光计算占比为97%左右  
推理结果的部分截图在./docs/推理结果 中  