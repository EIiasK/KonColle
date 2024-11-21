# 导入必要的库
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import DetrForObjectDetection, AutoConfig, logging
from torchvision.transforms import transforms
from torchvision.datasets import CocoDetection
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import copy
from PIL import Image

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
logging.set_verbosity_error()

# ==================== 自定义数据集类 ====================

def custom_collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return torch.stack(images, dim=0), targets

def process_annotations(target):
    """
    分离主标签和副属性。
    如果没有副属性，则创建一个空字典。
    """
    main_label = target["category_id"]
    sub_attributes = target.get("attributes", {"difficulty": "unknown", "confidence": 1.0})
    return main_label, sub_attributes


def get_image_categories(base_dir):
    """
    递归扫描目录，提取所有包含图片的子目录，并以子目录名称作为类别名。
    :param base_dir: 根目录路径
    :return: 一个字典 {类别名: 子目录路径}
    """
    categories = {}

    for root, dirs, files in os.walk(base_dir):
        # 检查当前目录下是否存在图片文件
        image_files = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        if image_files:  # 如果找到图片文件，将当前目录添加为类别
            relative_path = os.path.relpath(root, start=base_dir)
            categories[relative_path] = root

    return categories


class CustomCocoDataset(CocoDetection):
    def __init__(self, root, annotation_file, transform=None):
        """
        :param root: 图片的根目录
        :param annotation_file: COCO 格式标签文件
        :param transform: 数据增强方法
        """
        self.root = root
        super().__init__(root=root, annFile=annotation_file, transform=transform)

    def __getitem__(self, idx):
        """
        获取图片和标签，确保路径正确拼接，并处理副属性。
        """
        image, targets = super().__getitem__(idx)
        file_name = self.coco.loadImgs(targets[0]['image_id'])[0]['file_name']
        full_path = os.path.join(self.root, file_name)

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"未找到图像文件: {full_path}")

        # 加载图像
        image = Image.open(full_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # 分离主标签和副属性
        processed_targets = []
        for t in targets:
            main_label, sub_attributes = process_annotations(t)
            processed_targets.append({"label": main_label, "attributes": sub_attributes})

        return image, processed_targets


def main():
    # ==================== 参数配置 ====================
    base_dir = r"D:\Programming\Project\github\KonColle\Datasets\images"
    dataset_paths = get_image_categories(base_dir)

    print("自动生成的分类路径:")
    for category, path in dataset_paths.items():
        print(f"类别: {category}, 路径: {path}")

    annotation_file = r"D:\Programming\Project\github\KonColle\Datasets\annotations\instances_Train_fixed.json"
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

    dataset_paths_root = r"D:\Programming\Project\github\KonColle\Datasets\images"
    train_dataset = CustomCocoDataset(dataset_paths_root, annotation_file, transform=train_transforms)
    val_dataset = CustomCocoDataset(dataset_paths_root, annotation_file, transform=val_transforms)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=custom_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=custom_collate_fn
    )

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
            running_corrects = 0
            running_total = 0
            progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"{phase} Phase", ncols=100)

            for batch_idx, (images, targets) in progress_bar:
                images = torch.stack([img.to(device) for img in images])

                # 获取模型输出
                with autocast(device_type=device.type):
                    outputs = model(images)

                # 提取 labels
                num_queries = outputs.logits.shape[1]
                batch_labels = []
                for target_batch in targets:
                    query_labels = [0] * num_queries  # 初始化为背景类
                    for i, obj in enumerate(target_batch):
                        if i < num_queries:
                            query_labels[i] = obj["label"]  # 分配标签
                    batch_labels.append(query_labels)
                labels = torch.tensor(batch_labels, dtype=torch.long).to(device)

                # 计算损失
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    loss = criterion(outputs.logits.view(-1, outputs.logits.shape[-1]),
                                     labels.view(-1))  # 调整 logits 和 labels 的形状
                    if phase == "train":
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                # 计算准确率
                preds = torch.argmax(outputs.logits, dim=-1)  # 获取预测结果
                correct = 0
                total = 0
                for pred, label in zip(preds, labels):
                    for i in range(num_queries):
                        if label[i] != 0:  # 忽略背景类
                            total += 1
                            if pred[i] == label[i]:
                                correct += 1
                batch_accuracy = correct / total if total > 0 else 0  # 避免除以零

                running_corrects += correct
                running_total += total

                # 更新累计损失
                running_loss += loss.item()

                # 更新进度条显示内容
                progress_bar.set_postfix({
                    "Loss": f"{running_loss / ((batch_idx + 1) * batch_size):.4f}",
                    "Batch Acc": f"{batch_accuracy:.4f}"  # 当前批次准确率
                })

            # 计算阶段损失和准确率
            epoch_loss = running_loss / len(dataloader.dataset)
            if running_total > 0:
                epoch_accuracy = running_corrects / running_total
            else:
                epoch_accuracy = 0.0  # 避免除以零

            print(f"{phase} Loss: {epoch_loss:.4f}")
            print(f"{phase} Accuracy: {epoch_accuracy:.4f}")

            # 保存最佳模型
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

