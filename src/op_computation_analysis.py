import torch
import torch.nn as nn
import math


################### main calculate function ###################
def calculate_linear_macs(model, dummy_input):
    """
    计算 PyTorch 模型中所有线性计算算子（Conv 和 Linear）的计算量 (MACs)。
    
    参数:
        model: torch.nn.Module, 需要统计的模型
        dummy_input: torch.Tensor, 形状与模型实际输入一致的伪张量
        
    返回:
        total_macs: int, 总计算量
        macs_dict: dict, 记录每个具体算子名称及其计算量的字典
    """
    assert dummy_input.shape[0] ==1 # 确保batch size 为1 
    macs_dict = {}
    hooks = []
    
    # 建立一个模块到名称的映射，方便最后输出字典时使用人类可读的层名称
    module_to_name = {m: n for n, m in model.named_modules()}

    def linear_hook(module, input, output):
        # 线性层计算量: 输出元素总数 * 输入特征数
        # output shape: [batch_size, ..., out_features]
        out_elements = output.numel()
        macs = out_elements * module.in_features
        name = module_to_name[module]
        macs_dict[name] = macs

    def conv_hook(module, input, output):
        # 卷积层计算量: 输出特征图元素总数 * 每次卷积核滑动的乘加次数
        # 每次滑动的 MACs = (in_channels // groups) * kernel_size_1 * ... * kernel_size_n
        
        out_elements = output.numel()
        kernel_ops = (module.in_channels // module.groups) * math.prod(module.kernel_size)
        macs = out_elements * kernel_ops
        name = module_to_name[module]
        macs_dict[name] = macs

  

    # 1. 为目标算子注册 forward hook
    for module in model.modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))
        elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            hooks.append(module.register_forward_hook(conv_hook))

    # 2. 运行一次前向传播以触发 hooks
    model.eval()
    with torch.no_grad():
        # 确保输入和模型在同一个设备上
        device = next(model.parameters()).device
        model(dummy_input.to(device))

    # 3. 移除 hooks，避免对模型后续使用造成副作用
    for h in hooks:
        h.remove()

    # 4. 汇总总计算量
    total_macs = sum(macs_dict.values())

    return total_macs, macs_dict
################# main calculate function end #################
# ==========================================
# 测试用例
# ==========================================

def test_simpleNet():
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1,   groups=4)
            self.fc1 = nn.Linear(32 * 8 * 8, 128)
            self.fc2 = nn.Linear(128, 10)
        def forward(self, x):
            x = self.conv1(x)
            x = nn.functional.max_pool2d(x, 2) # [1, 16, 16, 16]
            x = self.conv2(x)
            x = nn.functional.max_pool2d(x, 2) # [1, 32, 8, 8]
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = self.fc2(x)
            return x

    model = SimpleNet()
    #model = models.resnet50(weights=None)
    # 假设输入是一个 Batch Size 为 1 的 32x32 RGB 图像
    dummy_input = torch.randn(1, 3, 32, 32) 

    # 调用计算函数
    total, details = calculate_linear_macs(model, dummy_input)

    # 打印结果

    print(f"=== 模型总计算量 (MACs) ===\n{total:,}")
    print("=== 各算子计算量详情 ===")
    for layer_name, macs in details.items():
        print(f"{layer_name:<10}: {macs:,} MACs")
    print('='*100)

def test_simpleNet2():
    class FCModel(nn.Module):
        def __init__(self):
            super(FCModel, self).__init__()
            self.fc1 = nn.Linear(1024, 1024)  
            self.fc2 = nn.Linear(1024, 512)  
            self.fc3 = nn.Linear(512, 100)    

        def forward(self, x):
            x = x.view(-1, )  
            x = torch.relu(self.fc1(x))  
            x = torch.relu(self.fc2(x))  
            x = self.fc3(x) 
            return x
    model = FCModel()
 
    dummy_input = torch.randn(1, 32,32) 

    # 调用计算函数
    total, details = calculate_linear_macs(model, dummy_input)

    # 打印结果
    print(f"=== 模型总计算量 (MACs) ===\n{total:,}")
    print("=== 各算子计算量详情 ===")
    for layer_name, macs in details.items():
        print(f"{layer_name:<10}: {macs:,} MACs")
    print('='*100)

def test_resnet50():

    from torchvision.models import resnet50
    model = resnet50(weights=None)
    dummy_input = torch.randn(1, 3, 224, 224) 

    # 调用计算函数
    total, details = calculate_linear_macs(model, dummy_input)

    # 打印结果
    
    print(f"=== 模型总计算量 (MACs) ===\n{total:,}")
    print("=== 各算子计算量详情 ===")
    for layer_name, macs in details.items():
        print(f"{layer_name:<10}: {macs:,} MACs")
    print('='*100)
    
if __name__ == "__main__":
    test_simpleNet()
    test_simpleNet2()
    test_resnet50()

