# 导入必要的库
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import DetrForObjectDetection, AutoConfig
from torchvision.transforms import transforms
from torchvision.datasets import CocoDetection
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import copy

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 0: 全部信息, 1: 信息+警告, 2: 仅错误, 3: 严重错误
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # 禁用 OneDNN

# ==================== 自定义数据集类 ====================
class CustomCocoDataset(CocoDetection):
    def __init__(self, dataset_paths, annotation_file, transform=None):
        self.dataset_paths = dataset_paths
        self.annotation_file = annotation_file
        self.transform = transform
        super(CustomCocoDataset, self).__init__(root=list(dataset_paths.values())[0], annFile=annotation_file, transform=transform)

    def __getitem__(self, idx):
        image, target = super(CustomCocoDataset, self).__getitem__(idx)
        return image, target


def main():
    # ==================== 参数配置 ====================
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

    dataset_paths = {}
    for base_dir in base_dirs:
        for entry in os.scandir(base_dir):
            if entry.is_dir():
                folder_name = os.path.basename(entry.path)
                dataset_paths[folder_name] = entry.path

    print("自动生成的分类路径:")
    for category, path in dataset_paths.items():
        print(f"类别: {category}, 路径: {path}")

    annotation_file = r"D:\Programming\Project\github\KonColle\Datasets\annotations\instances_Train.json"
    img_width, img_height = 1111, 667
    batch_size = 8
    epochs = 30
    learning_rate = 1e-5
    model_save_path = r"D:\Programming\Project\github\KonColle\KC\Models\detr_model.pth"

    # ==================== 数据预处理和加载 ====================
    train_transforms = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = CustomCocoDataset(dataset_paths, annotation_file, transform=train_transforms)
    val_dataset = CustomCocoDataset(dataset_paths, annotation_file, transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # ==================== 构建模型 ====================
    config_path = r"D:\Programming\Project\github\KonColle\KC\Models\DETR\config.json"
    weight_path = r"D:\Programming\Project\github\KonColle\KC\Models\DETR\pytorch_model.bin"
    config = AutoConfig.from_pretrained(config_path)

    model = DetrForObjectDetection.from_pretrained(
        weight_path,
        config=config
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("模型加载成功")

    # ==================== 定义损失函数和优化器 ====================
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3)
    scaler = GradScaler()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float("inf")

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        print("-" * 10)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"{phase} Phase", ncols=100)

            for batch_idx, (images, targets) in progress_bar:
                images = [img.to(device) for img in images]
                targets = [{k: torch.tensor(v).to(device) for k, v in t.items()} for t in targets]

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    with autocast():
                        outputs = model(images)
                        loss = criterion(outputs, targets)
                    if phase == "train":
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                running_loss += loss.item()
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
    torch.save(model.state_dict(), model_save_path)
    print(f"最终模型已保存至: {model_save_path}")


if __name__ == "__main__":
    main()
