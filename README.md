├── data/                 # 数据存储目录，不要上传原始大文件到 Git，此文件夹不会被上传到git本地与github
│   ├── raw/              # 原始数据
│   ├── processed/        # 处理后的数据
├── docs/                 # 项目文档，找到的资料可以塞这里
├── models/               # 存放训练好的权重文件 (.ckpt, .pth, .onnx)（默认不上传到git本地与github，如果想上传，起一个清晰的名字后参照，gitinore中的方法进行保留）
├── src/                  # 代码         
│   └── train.py          
├── tests/                # 测试代码
│   ├── test_conv.py
│   └── test_gemm.py
├── configs/              # 配置文件（YAML/JSON格式，方便修改参数）
│   └── config.yaml
├── README.md             
├── .gitignore            # 忽略不需要上传的文件