├── data/                 # 数据存储目录  (tips：此文件夹在 .gitignore 中配置，不会被上传至 Git 本地仓库及 GitHub）
│   ├── raw/              # 原始数据
│   └── processed/        # 处理后的数据
│      
├── docs/                 # 项目文档
│   └── ...               # 存放比赛资料、算法推导及硬件方案文档
│
├── models/               # 模型权重存放区
│   └── *.ckpt/*.pth      # 存放训练好的权重文件
│       (注：默认由 .gitignore 忽略。若需上传特定文件，请在 .gitignore 中配置白名单)
│
├── src/                  # 核心源代码
│   └── train.py          
│
├── tests/                # 测试代码
│   └── test_gemm.py     
│
├── configs/              # 配置文件
│   └── config.yaml       # YAML/JSON 格式参数配置
│
├── README.md             # 项目简介与使用指南
└── .gitignore            # Git 忽略规则配置