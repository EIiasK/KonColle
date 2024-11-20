# 导入必要的库（保持不变）
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import pandas as pd
import os
import math
from PIL import Image
import datetime
import copy
import numpy as np
from torch.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torchvision.datasets import CocoDetection
# ==================== 数据集定义 ====================

class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.data = dataframe.reset_index(drop=True)
        self.transform = transform
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(sorted(self.data['class'].unique()))}
        self.idx_to_class = {idx: cls_name for cls_name, idx in self.class_to_idx.items()}
        self.data['class_idx'] = self.data['class'].map(self.class_to_idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.loc[idx, 'filename']
        label = self.data.loc[idx, 'class_idx']
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# ==================== 参数配置 ====================

# 图像的宽度和高度
img_width, img_height = 1111, 667

# 训练的批次大小和训练轮数
batch_size = 16
epochs = 30

# 定义数据集路径和对应的类别标签
dataset_paths = {
    'main_menu': r'D:\Programming\Project\github\KonColle\Datasets\images\main_menu',
    'supply': r'D:\Programming\Project\github\KonColle\Datasets\images\supply',
    'mission_select': r'D:\Programming\Project\github\KonColle\Datasets\images\mission_select',
    'map_1': r'D:\Programming\Project\github\KonColle\Datasets\images\waters\map_1',
    'map_5': r'D:\Programming\Project\github\KonColle\Datasets\images\waters\map_5',
    'map_5_info': r'D:\Programming\Project\github\KonColle\Datasets\images\waters\map_5\map_info',
    'map_5_fleet': r'D:\Programming\Project\github\KonColle\Datasets\images\waters\map_5\map_fleet',
    'advance_on': r'D:\Programming\Project\github\KonColle\Datasets\images\waters\in_map\advance_on',
    'combat_rating': r'D:\Programming\Project\github\KonColle\Datasets\images\waters\in_map\combat_rating',
    'combat_result': r'D:\Programming\Project\github\KonColle\Datasets\images\waters\in_map\combat_result',
    'navigation': r'D:\Programming\Project\github\KonColle\Datasets\images\waters\in_map\navigation',
    # 可以添加更多类别
}

if __name__ == '__main__':
    # 创建列表来存储文件路径和标签
    file_paths = []
    labels = []

    for label, directory in dataset_paths.items():
        # 遍历指定目录（不包括子目录）
        for filename in os.listdir(directory):  # 只读取当前目录
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # 筛选图片文件
                file_paths.append(os.path.join(directory, filename))
                labels.append(label)

    # 创建 DataFrame
    data = pd.DataFrame({'filename': file_paths, 'class': labels})

    # 打印每个类别的样本数量
    print("每个类别的样本数量：")
    print(data['class'].value_counts())

    # 类别数量
    num_classes = len(data['class'].unique())

    # 划分训练和验证集（按照 80% 训练，20% 验证）
    train_data, val_data = train_test_split(data, test_size=0.2, stratify=data['class'], random_state=42)

    # 打印训练集每个类别的样本数量
    print("\n训练集每个类别的样本数量：")
    print(train_data['class'].value_counts())

    # 打印验证集每个类别的样本数量
    print("\n验证集每个类别的样本数量：")
    print(val_data['class'].value_counts())
    # ==================== 数据预处理和增强 ====================

    # 定义数据增强和预处理
    train_transforms = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.RandomAdjustSharpness(sharpness_factor=2),
        transforms.ColorJitter(brightness=0.1),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet 的均值和标准差
                             std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 创建数据集实例
    train_dataset = CustomDataset(train_data, transform=train_transforms)
    val_dataset = CustomDataset(val_data, transform=val_transforms)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 检查类别索引映射
    print("类别索引映射:", train_dataset.class_to_idx)

    # ==================== 构建模型 ====================

    # 加载预训练的 ResNet50 模型
    base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    # 修改最后的全连接层
    num_ftrs = base_model.fc.in_features

    # 添加自定义的顶层网络
    base_model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, num_classes)
    )

    # 将模型移动到 GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = base_model.to(device)

    # 冻结部分层，解冻后面的层进行微调
    for param in list(model.parameters())[:-10]:
        param.requires_grad = False
    for param in list(model.parameters())[-10:]:
        param.requires_grad = True

    # ==================== 定义损失函数和优化器 ====================

    # 定义损失函数
    criterion = nn.CrossEntropyLoss()

    # 定义优化器
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)

    # 定义学习率调度器（可选）
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

    # 定义混合精度缩放器
    scaler = GradScaler()

    # ==================== 设置日志和模型保存路径 ====================

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(log_dir, exist_ok=True)

    model_save_path = r'D:\Programming\Project\github\KonColle\KC\Models\model.pth'
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # ==================== 训练模型 ====================

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        print('-' * 10)

        # 每个 epoch 包含一个训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"{phase} Phase", ncols=100)

            for batch_idx, (inputs, labels) in progress_bar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    if phase == 'train':
                        with autocast(device_type='cuda'):
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        with torch.no_grad():
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                # 统计损失和准确率
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # 计算当前的平均损失和准确率
                current_loss = running_loss / ((batch_idx + 1) * dataloader.batch_size)
                current_acc = running_corrects.double() / ((batch_idx + 1) * dataloader.batch_size)

                # 更新进度条的信息
                progress_bar.set_postfix({
                    'Loss': f'{current_loss:.4f}',
                    'Acc': f'{current_acc:.4f}'
                })

            # 计算整个 epoch 的损失和准确率
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f'\n{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 深度复制模型（如果在验证阶段取得了更好的结果）
            if phase == 'val':
                scheduler.step(epoch_loss)  # 更新学习率
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    # 保存最佳模型
                    torch.save(model.state_dict(), model_save_path)
                    print(f'最佳模型已更新并保存到: {model_save_path}')

        print()

    print('训练完成')
    print(f'最佳验证损失: {best_loss:.4f}')

    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)

    # ==================== 保存最终模型 ====================

    # 保存模型到指定路径
    torch.save(model.state_dict(), model_save_path)
    print("模型已成功保存到:", model_save_path)

    # 打印模型结构
    print(model)
