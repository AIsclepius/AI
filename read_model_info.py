import torch
import os

def read_model_info(model_path):
    # 检查文件是否存在
    if not os.path.exists(model_path):
        print(f"错误：文件 {model_path} 不存在")
        return
    
    try:
        # 加载模型文件
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))  # 加载到CPU
        
        print(f"成功加载模型文件: {model_path}")
        print("\n文件包含的键值对:")
        for key in checkpoint.keys():
            print(f"- {key}")
        
        # 如果包含模型状态字典，查看模型结构信息
        if 'state_dict' in checkpoint:
            print("\n模型状态字典包含的层信息:")
            state_dict = checkpoint['state_dict']
            for name, param in list(state_dict.items())[:10]:  # 只显示前10层
                print(f"- {name}: 形状 {param.shape}")
            
            # 计算总参数数量
            total_params = sum(p.numel() for p in state_dict.values())
            print(f"\n模型总参数数量: {total_params:,}")
        
        # 如果包含优化器信息
        if 'optimizer' in checkpoint:
            print("\n优化器信息:")
            print(f"优化器类型: {type(checkpoint['optimizer'])}")
            
        # 如果包含训练参数
        if 'epoch' in checkpoint:
            print(f"\n训练到的 epoch: {checkpoint['epoch']}")
            
    except Exception as e:
        print(f"读取模型文件时出错: {str(e)}")

if __name__ == "__main__":
    # 模型文件路径
    model_path = r"D:\AIsclepius\pneumonia_classifier\pretrained\model.pth.tar"
    read_model_info(model_path)
    
