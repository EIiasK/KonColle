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
from torchvision.ops import box_iou
from pycocotools.coco import COCO

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
logging.set_verbosity_error()

# ==================== 自定义数据集类 ====================

def custom_collate_fn(batch):
    images = [item[0] for item in batch]
    detection_targets = [item[1] for item in batch]
    classification_targets = [item[2] for item in batch]
    # 将classification_targets从列表转换为张量
    classification_targets = torch.tensor([t.item() for t in classification_targets])
    return images, detection_targets, classification_targets


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
        super().__init__(root=root, annFile=annotation_file)
        self.transform = transform

        # 获取所有类别的名称和ID，用于图像分类标签
        self.coco = COCO(annotation_file)
        cats = self.coco.loadCats(self.coco.getCatIds())
        self.class_id_to_name = {cat['id']: cat['name'] for cat in cats}
        self.class_name_to_id = {v: k for k, v in self.class_id_to_name.items()}

    def __getitem__(self, idx):
        img, ann = super().__getitem__(idx)
        if self.transform is not None:
            img = self.transform(img)

        # 获取目标检测标签
        target = {}
        boxes = []
        labels = []
        area = []
        iscrowd = []
        for obj in ann:
            xmin = obj['bbox'][0]
            ymin = obj['bbox'][1]
            width = obj['bbox'][2]
            height = obj['bbox'][3]
            boxes.append([xmin, ymin, xmin + width, ymin + height])
            labels.append(obj['category_id'])
            area.append(obj['area'])
            iscrowd.append(obj.get('iscrowd', 0))
        target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
        target['class_labels'] = torch.as_tensor(labels, dtype=torch.int64)  # 已修改键名
        target['image_id'] = torch.tensor([idx])
        target['area'] = torch.as_tensor(area, dtype=torch.float32)
        target['iscrowd'] = torch.as_tensor(iscrowd, dtype=torch.int64)

        # 获取图像分类标签（假设每张图像的分类标签为其中出现次数最多的类别）
        classification_label = torch.mode(target['class_labels'])[0]

        return img, target, classification_label



def main():
    # ==================== 参数配置 ====================
    base_dir = r"D:\Programming\Project\github\KonColle\Datasets\images"
    dataset_paths = get_image_categories(base_dir)

    print("自动生成的分类路径:")
    for category, path in dataset_paths.items():
        print(f"类别: {category}, 路径: {path}")

    annotation_file = r"D:\Programming\Project\github\KonColle\Datasets\annotations\instances_Train_fixed.json"
    img_width, img_height = 1111, 667
    batch_size = 4  # 减小batch_size以适应多任务训练的显存需求
    epochs = 30
    learning_rate = 1e-5
    model_save_path = r"D:\Programming\Project\github\KonColle\KC\Models\detr_multitask_model.pth"

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

    # 添加图像分类的全连接层
    num_classes = len(train_dataset.class_id_to_name) + 1  # 加1表示背景类
    model.classification_head = nn.Linear(config.hidden_size, num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("模型加载成功")

    # ==================== 定义损失函数和优化器 ====================
    detection_criterion = nn.CrossEntropyLoss()
    classification_criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3)
    scaler = GradScaler("cuda")

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
            running_classification_loss = 0.0  # 新增：累计分类损失
            running_detection_loss = 0.0  # 新增：累计检测损失
            running_corrects = 0
            running_total = 0
            progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"{phase} Phase")

            for batch_idx, (images, detection_targets, classification_targets) in progress_bar:
                images = torch.stack([img.to(device) for img in images])
                detection_targets = [{k: v.to(device) for k, v in t.items()} for t in detection_targets]
                classification_targets = classification_targets.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    with autocast(device_type=device.type):
                        outputs = model(images, labels=detection_targets)

                        # 分类任务的输出
                        encoder_outputs = outputs.encoder_last_hidden_state  # [batch_size, num_patches, hidden_dim]
                        pooled_output = encoder_outputs.mean(dim=1)  # [batch_size, hidden_dim]
                        classification_logits = model.classification_head(pooled_output)

                        # 计算分类损失
                        classification_loss = classification_criterion(classification_logits, classification_targets)

                        # 计算检测损失
                        detection_loss = outputs.loss

                        # 总损失（可以根据需要调整权重）
                        total_loss = classification_loss + detection_loss

                    if phase == "train":
                        scaler.scale(total_loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                # 更新累计损失
                batch_size_current = images.size(0)
                running_loss += total_loss.item() * batch_size_current
                running_classification_loss += classification_loss.item() * batch_size_current  # 累计分类损失
                running_detection_loss += detection_loss.item() * batch_size_current  # 累计检测损失

                # 计算分类准确率
                _, preds = torch.max(classification_logits, 1)
                running_corrects += torch.sum(preds == classification_targets.data)
                running_total += batch_size_current

                # 更新进度条显示内容
                progress_bar.set_postfix({
                    "Total Loss": f"{running_loss / running_total:.4f}",
                    "Cls Loss": f"{running_classification_loss / running_total:.4f}",
                    "Det Loss": f"{running_detection_loss / running_total:.4f}",
                    "Acc": f"{running_corrects.double() / running_total:.4f}"
                })

            # 计算阶段损失和准确率
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_cls_loss = running_classification_loss / len(dataloader.dataset)
            epoch_det_loss = running_detection_loss / len(dataloader.dataset)
            epoch_accuracy = running_corrects.double() / len(dataloader.dataset)

            print(f"{phase} Total Loss: {epoch_loss:.4f}")
            print(f"{phase} Classification Loss: {epoch_cls_loss:.4f}")
            print(f"{phase} Detection Loss: {epoch_det_loss:.4f}")
            print(f"{phase} Accuracy: {epoch_accuracy:.4f}")

            # 学习率调整
            if phase == "val":
                scheduler.step(epoch_loss)

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


