# 导入必要的库
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import DetrForObjectDetection, AutoConfig  # 加载 DETR 模型
from torchvision.transforms import transforms  # 图像预处理
from torchvision.datasets import CocoDetection  # 加载 COCO 格式数据集
from torch.optim import AdamW  # 优化器
from torch.optim.lr_scheduler import ReduceLROnPlateau  # 学习率调度器
from torch.amp import autocast, GradScaler  # 混合精度
from tqdm import tqdm  # 进度条
import copy
import datetime
import timm

# ==================== 参数配置 ====================

# 定义多个相互包含的文件夹路径（用来动态生成类别路径）
base_dirs = [
    r'D:\Programming\Project\github\KonColle\Datasets\images\main_menu',
    r'D:\Programming\Project\github\KonColle\Datasets\images\supply',
    r'D:\Programming\Project\github\KonColle\Datasets\images\mission_select',
    r'D:\Programming\Project\github\KonColle\Datasets\images\waters\map_1',
    r'D:\Programming\Project\github\KonColle\Datasets\images\waters\map_5',
    r'D:\Programming\Project\github\KonColle\Datasets\images\waters\map_5\map_5_info',
    r'D:\Programming\Project\github\KonColle\Datasets\images\waters\map_5\map_5_fleet',
    r'D:\Programming\Project\github\KonColle\Datasets\images\waters\in_map\advance_on',
    r'D:\Programming\Project\github\KonColle\Datasets\images\waters\in_map\combat_rating',
    r'D:\Programming\Project\github\KonColle\Datasets\images\waters\in_map\combat_result',
    r'D:\Programming\Project\github\KonColle\Datasets\images\waters\in_map\navigation',
]

# 创建字典用于存储类别路径（只存储首层文件夹图片）
dataset_paths = {}

# 遍历每个基础路径
for base_dir in base_dirs:
    for entry in os.scandir(base_dir):  # 使用 scandir 遍历首层目录
        if entry.is_dir():  # 确保只处理文件夹
            folder_name = os.path.basename(entry.path)  # 文件夹名作为类别名
            dataset_paths[folder_name] = entry.path

# 输出生成的类别路径
print("自动生成的分类路径:")
for category, path in dataset_paths.items():
    print(f"类别: {category}, 路径: {path}")

# COCO 格式的标签文件路径
annotation_file = r"D:\Programming\Project\github\KonColle\Datasets\annotations\instances_Train.json"

# 图像尺寸和训练参数
img_width, img_height = 1111, 667  # 目标图像尺寸
batch_size = 8
epochs = 30
learning_rate = 1e-5

# 模型保存路径
model_save_path = r"D:\Programming\Project\github\KonColle\KC\Models\detr_model.pth"

# ==================== 数据预处理和加载 ====================

# 定义训练数据的预处理和增强
train_transforms = transforms.Compose([
    transforms.Resize((img_height, img_width)),  # 调整图像尺寸
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),  # 随机调整亮度、对比度、饱和度
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
])

# 定义验证数据的预处理（无增强）
val_transforms = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 自定义 COCO 数据集类，加载图像和标签
class CustomCocoDataset(CocoDetection):
    def __init__(self, dataset_paths, annotation_file, transform=None):
        self.dataset_paths = dataset_paths  # 类别路径
        self.annotation_file = annotation_file  # COCO 标签文件
        self.transform = transform  # 图像预处理
        super(CustomCocoDataset, self).__init__(root=list(dataset_paths.values())[0], annFile=annotation_file, transform=transform)

    def __getitem__(self, idx):
        # 调用父类方法获取图像和标签
        image, target = super(CustomCocoDataset, self).__getitem__(idx)
        return image, target

# 初始化数据集和数据加载器
train_dataset = CustomCocoDataset(dataset_paths, annotation_file, transform=train_transforms)
val_dataset = CustomCocoDataset(dataset_paths, annotation_file, transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# ==================== 构建模型 ====================


# 本地配置和权重路径
config_path = r"D:\Programming\Project\github\KonColle\KC\Models\DETR\config.json"
weight_path = r"D:\Programming\Project\github\KonColle\KC\Models\DETR\pytorch_model.bin"  # 或 pytorch_model.bin

# 加载配置文件
config = AutoConfig.from_pretrained(config_path)

# 加载模型权重
model = DetrForObjectDetection.from_pretrained(
    weight_path,
    config=config
)

# 移动模型到 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("模型加载成功")

# ==================== 定义损失函数和优化器 ====================

criterion = nn.CrossEntropyLoss()  # 分类损失函数
optimizer = AdamW(model.parameters(), lr=learning_rate)  # AdamW 优化器
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3)  # 学习率调度器
scaler = GradScaler()  # 混合精度缩放器

# ==================== 训练模型 ====================

# 保存最佳模型
best_model_wts = copy.deepcopy(model.state_dict())
best_loss = float("inf")

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    print("-" * 10)

    for phase in ["train", "val"]:
        if phase == "train":
            model.train()  # 训练模式
            dataloader = train_loader
        else:
            model.eval()  # 验证模式
            dataloader = val_loader

        running_loss = 0.0

        # 显示进度条
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"{phase} Phase", ncols=100)

        for batch_idx, (images, targets) in progress_bar:
            images = [img.to(device) for img in images]
            targets = [{k: torch.tensor(v).to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == "train"):
                with autocast():  # 混合精度训练
                    outputs = model(images)
                    loss = criterion(outputs, targets)  # 自定义损失函数
                if phase == "train":
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

            running_loss += loss.item()

            # 更新进度条信息
            progress_bar.set_postfix({"Loss": f"{running_loss / ((batch_idx + 1) * batch_size):.4f}"})

        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"{phase} Loss: {epoch_loss:.4f}")

        if phase == "val" and epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), model_save_path)
            print(f"最佳模型已保存至: {model_save_path}")

print("训练完成")
model.load_state_dict(best_model_wts)

# ==================== 保存最终模型 ====================

torch.save(model.state_dict(), model_save_path)
print(f"最终模型已保存至: {model_save_path}")
