import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms
from torchvision.datasets import CocoDetection
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import copy
from collections import Counter
from PIL import Image
from torchvision.ops import box_iou
from pycocotools.coco import COCO
from sklearn.model_selection import train_test_split
import json

# 添加 YOLOv8 的导入
from ultralytics import YOLO
from ultralytics.utils.loss import v8DetectionLoss

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ==================== 自定义函数 ====================

def get_class_distribution(dataset):
    class_counts = Counter()
    for _, _, classification_label in dataset:
        class_counts[int(classification_label)] += 1
    return class_counts

def custom_collate_fn(batch):
    images = [item[0] for item in batch]
    detection_targets = [item[1] for item in batch]
    classification_targets = [item[2] for item in batch]
    classification_targets = torch.tensor(classification_targets)
    return images, detection_targets, classification_targets

def get_category_image_paths(base_dir):
    category_image_paths = {}

    for root, dirs, files in os.walk(base_dir):
        # 检查当前目录下是否存在图片文件
        image_files = [f for f in files if f.lower().endswith(
            ('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        if image_files:
            # 计算相对于 base_dir 的相对路径作为类别名称
            category = os.path.relpath(root, base_dir)
            # 构建图片文件的相对路径列表
            image_paths = [os.path.join(category, f) for f in image_files]
            category_image_paths[category] = image_paths
            print(f"类别 '{category}' 的图片数量：{len(image_paths)}")
        else:
            print(f"类别 '{os.path.relpath(root, base_dir)}' 中没有找到图片文件。")
    return category_image_paths

def split_dataset(category_image_paths, test_size=0.2, random_state=42):
    train_images = []
    val_images = []
    for category, image_paths in category_image_paths.items():
        if not image_paths:
            print(f"类别 '{category}' 中没有图片可供划分。")
            continue
        train_imgs, val_imgs = train_test_split(
            image_paths, test_size=test_size, random_state=random_state
        )
        train_images.extend(train_imgs)
        val_images.extend(val_imgs)
    return train_images, val_images

def filter_coco_annotations(annotation_file, image_files, output_file):
    with open(annotation_file, 'r') as f:
        coco_data = json.load(f)

    # 获取图片文件名到图片ID的映射
    filename_to_id = {
        img['file_name']: img['id'] for img in coco_data['images']}

    # 过滤图片和注释
    selected_image_ids = set()
    selected_images = []
    for img in coco_data['images']:
        if img['file_name'] in image_files:
            selected_images.append(img)
            selected_image_ids.add(img['id'])

    selected_annotations = [
        ann for ann in coco_data['annotations'] if ann['image_id'] in selected_image_ids
    ]

    # 构建新的COCO标注数据
    new_coco_data = {
        'info': coco_data.get('info', {}),
        'licenses': coco_data.get('licenses', []),
        'images': selected_images,
        'annotations': selected_annotations,
        'categories': coco_data.get('categories', [])
    }

    # 保存新的标注文件
    with open(output_file, 'w') as f:
        json.dump(new_coco_data, f)

# 新增函数：将边界框从 [xmin, ymin, xmax, ymax] 转换为 [x_center, y_center, width, height]

def xyxy2xywh(boxes):
    x_center = (boxes[:, 0] + boxes[:, 2]) / 2
    y_center = (boxes[:, 1] + boxes[:, 3]) / 2
    width = boxes[:, 2] - boxes[:, 0]
    height = boxes[:, 3] - boxes[:, 1]
    return torch.stack([x_center, y_center, width, height], dim=1)

# ==================== 自定义数据集类 ====================

class CustomCocoDataset(CocoDetection):
    def __init__(self, root, annotation_file, transform=None, image_files=None):
        super().__init__(root=root, annFile=annotation_file)
        self.transform = transform

        # 获取所有类别的名称和ID，用于图像分类标签
        category_names = [d for d in self.get_all_categories(root)]
        self.class_name_to_id = {name: idx for idx,
                                 name in enumerate(sorted(category_names))}
        self.class_id_to_name = {idx: name for name,
                                 idx in self.class_name_to_id.items()}

        # 如果提供了 image_files，过滤 self.ids
        if image_files is not None:
            filename_to_id = {
                img_info['file_name']: img_id for img_id, img_info in self.coco.imgs.items()}
            self.ids = [filename_to_id[fname]
                        for fname in image_files if fname in filename_to_id]

    def get_all_categories(self, root):
        categories = set()
        for root_dir, dirs, files in os.walk(root):
            image_files = [f for f in files if f.lower().endswith(
                ('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
            if image_files:
                category = os.path.relpath(root_dir, root)
                categories.add(category)
        return categories

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
            labels.append(obj['category_id'])  # 使用原始的 category_id
            area.append(obj['area'])
            iscrowd.append(obj.get('iscrowd', 0))
        target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
        target['class_labels'] = torch.as_tensor(
            labels, dtype=torch.int64)  # 修改键名
        target['image_id'] = torch.tensor([self.ids[idx]])
        target['area'] = torch.as_tensor(area, dtype=torch.float32)
        target['iscrowd'] = torch.as_tensor(iscrowd, dtype=torch.int64)

        # 获取图像分类标签（基于文件夹名）
        img_info = self.coco.imgs[self.ids[idx]]
        img_file_name = img_info['file_name']
        category_name = os.path.dirname(img_file_name)
        classification_label = self.class_name_to_id[category_name]
        return img, target, classification_label

# 修改后的自定义模型类，将分类头添加到 YOLOv8 模型上

class YOLOv8WithClassification(nn.Module):
    def __init__(self, yolo_model, num_classes):
        super(YOLOv8WithClassification, self).__init__()
        self.model = yolo_model.model  # 这是底层的 nn.Module
        self.num_classes = num_classes
        self.classification_head = None  # 分类头将在第一次前向传播时初始化
        self.features = None  # 用于存储中间特征

        # 注册前向钩子以获取特征
        self._register_hook()

        # 初始化 v8DetectionLoss
        self.loss_func = v8DetectionLoss(self.model)

    def _register_hook(self):
        # 注册一个前向钩子，在指定层捕获特征
        # 假设我们在检测头之前的层（即 self.model.model[-2]）捕获特征
        def hook(module, input, output):
            self.features = output

        self.model.model[-2].register_forward_hook(hook)

    def forward(self, x, targets=None):
        # 每次前向传播前重置特征
        self.features = None

        # 前向传播
        predictions = self.model(x)

        # 获取中间特征
        features = self.features
        if features is None:
            raise ValueError("未能捕获特征，请检查前向钩子的设置。")

        # 分类任务
        gap = torch.mean(features, dim=(2, 3))  # 全局平均池化
        if self.classification_head is None:
            # 初始化分类头
            feature_dim = gap.shape[1]
            self.classification_head = nn.Linear(
                feature_dim, self.num_classes).to(x.device)
        classification_logits = self.classification_head(gap)

        if self.training:
            # 计算检测损失
            detection_loss, _ = self.loss_func(predictions, {'batch_idx': targets['batch_idx'], 'cls': targets['cls'], 'bboxes': targets['bboxes']})
            return classification_logits, detection_loss
        else:
            return classification_logits, predictions

# ==================== 主函数 ====================

def main():
    # ==================== 参数配置 ====================
    base_dir = r"D:\Programming\Project\github\KonColle\Datasets\images"
    annotation_file = r"D:\Programming\Project\github\KonColle\Datasets\annotations\instances_Train_fixed.json"
    img_width, img_height = 640, 640  # YOLOv8 默认输入尺寸
    batch_size = 8
    epochs = 30
    learning_rate = 1e-4
    model_save_path = r"D:\Programming\Project\github\KonColle\KC\Models\yolov8_multitask_model.pth"
    log_dir = r"D:\Programming\Project\github\KonColle\KC\Logs"
    # 创建 TensorBoard 的 SummaryWriter
    writer = SummaryWriter(log_dir=log_dir)

    # ==================== 数据预处理和加载 ====================
    train_transforms = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    # 获取每个类别的图片列表
    category_image_paths = get_category_image_paths(base_dir)

    # 检查是否获取到了图片
    if not category_image_paths:
        print("未找到任何类别的图片文件。请检查 base_dir 路径和文件夹结构。")
        return

    # 划分训练集和验证集
    train_images, val_images = split_dataset(
        category_image_paths, test_size=0.2, random_state=42)

    # 生成训练集和验证集的COCO标注文件
    train_annotation_file = r"D:\Programming\Project\github\KonColle\Datasets\annotations\instances_train.json"
    val_annotation_file = r"D:\Programming\Project\github\KonColle\Datasets\annotations\instances_val.json"

    filter_coco_annotations(
        annotation_file, train_images, train_annotation_file)
    filter_coco_annotations(annotation_file, val_images, val_annotation_file)

    # 创建训练集和验证集的数据集对象
    train_dataset = CustomCocoDataset(
        base_dir, train_annotation_file, transform=train_transforms,
        image_files=train_images
    )
    val_dataset = CustomCocoDataset(
        base_dir, val_annotation_file, transform=val_transforms,
        image_files=val_images
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=custom_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=custom_collate_fn
    )

    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")

    train_class_distribution = get_class_distribution(train_dataset)
    print("训练集类别分布：", train_class_distribution)

    val_class_distribution = get_class_distribution(val_dataset)
    print("验证集类别分布：", val_class_distribution)

    # ==================== 构建模型 ====================
    # 加载预训练的 YOLOv8 模型
    yolov8_model = YOLO('yolov8n.pt')

    # 创建包含分类头的自定义模型
    num_classes = len(train_dataset.class_name_to_id)
    model = YOLOv8WithClassification(yolov8_model, num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("模型加载成功")

    # ==================== 定义损失函数和优化器 ====================
    classification_criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=3)
    scaler = torch.amp.GradScaler()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float("inf")

    # 训练和验证循环
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
            running_classification_loss = 0.0
            running_detection_loss = 0.0
            running_corrects = 0
            running_total = 0
            progress_bar = tqdm(enumerate(
                dataloader), total=len(dataloader), desc=f"{phase} Phase")

            for batch_idx, (images, detection_targets, classification_targets) in progress_bar:
                images = torch.stack([img.to(device) for img in images])
                classification_targets = classification_targets.to(device)

                # 准备检测目标
                target_list = []
                for i, target in enumerate(detection_targets):
                    boxes = target['boxes']  # [n,4]
                    labels = target['class_labels']  # [n]

                    # 规范化坐标
                    boxes[:, 0] /= img_width
                    boxes[:, 1] /= img_height
                    boxes[:, 2] /= img_width
                    boxes[:, 3] /= img_height

                    batch_idx_tensor = torch.full((len(labels), 1), i, dtype=torch.float32)
                    labels = labels.unsqueeze(1).float()
                    targets_per_image = torch.cat([batch_idx_tensor, labels, boxes], dim=1)  # [n,6]
                    target_list.append(targets_per_image)

                if target_list:
                    targets = torch.cat(target_list, dim=0).to(device)
                    batch_targets = {
                        'batch_idx': targets[:, 0].long(),
                        'cls': targets[:, 1].long(),
                        'bboxes': targets[:, 2:6]
                    }
                else:
                    batch_targets = {
                        'batch_idx': torch.tensor([], dtype=torch.int64).to(device),
                        'cls': torch.tensor([], dtype=torch.int64).to(device),
                        'bboxes': torch.tensor([], dtype=torch.float32).to(device)
                    }

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    with torch.autocast(device_type=device.type, dtype=torch.float16):
                        outputs = model(images, targets=batch_targets)
                        classification_logits, detection_loss = outputs

                        # 计算分类损失
                        classification_loss = classification_criterion(
                            classification_logits, classification_targets)

                        # 总损失
                        total_loss = classification_loss + detection_loss

                if phase == "train":
                    scaler.scale(total_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                # 更新累计损失
                batch_size_current = images.size(0)
                running_loss += total_loss.item() * batch_size_current
                running_classification_loss += classification_loss.item(
                ) * batch_size_current
                running_detection_loss += detection_loss.item(
                ) * batch_size_current

                # 计算分类准确率
                _, preds = torch.max(classification_logits, 1)
                running_corrects += torch.sum(
                    preds == classification_targets.data)
                running_total += batch_size_current

                # 每隔20个批次打印一次预测结果
                if batch_idx % 20 == 0:
                    print(f"真实标签: {classification_targets.cpu().numpy()}")
                    print(f"预测标签: {preds.cpu().numpy()}")

                # 更新进度条显示内容
                progress_bar.set_postfix({
                    "Total Loss": f"{running_loss / running_total:.4f}",
                    "Cls Loss": f"{running_classification_loss / running_total:.4f}",
                    "Det Loss": f"{running_detection_loss / running_total:.4f}",
                    "Acc": f"{(running_corrects.double() / running_total).item():.4f}"
                })

            # 计算阶段损失和准确率
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_cls_loss = running_classification_loss / \
                len(dataloader.dataset)
            epoch_det_loss = running_detection_loss / len(dataloader.dataset)
            epoch_accuracy = running_corrects.double() / len(dataloader.dataset)

            print(f"{phase} Total Loss: {epoch_loss:.4f}")
            print(f"{phase} Classification Loss: {epoch_cls_loss:.4f}")
            print(f"{phase} Detection Loss: {epoch_det_loss:.4f}")
            print(f"{phase} Accuracy: {epoch_accuracy:.4f}")

            # 将损失和准确率记录到 TensorBoard
            if phase == "train":
                writer.add_scalar('Loss/Train_Total', epoch_loss, epoch)
                writer.add_scalar(
                    'Loss/Train_Classification', epoch_cls_loss, epoch)
                writer.add_scalar(
                    'Loss/Train_Detection', epoch_det_loss, epoch)
                writer.add_scalar('Accuracy/Train', epoch_accuracy, epoch)
            else:
                writer.add_scalar('Loss/Val_Total', epoch_loss, epoch)
                writer.add_scalar(
                    'Loss/Val_Classification', epoch_cls_loss, epoch)
                writer.add_scalar('Loss/Val_Detection', epoch_det_loss, epoch)
                writer.add_scalar('Accuracy/Val', epoch_accuracy, epoch)

            # 学习率调整
            if phase == "val":
                scheduler.step(epoch_cls_loss)  # 监控分类损失

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

    writer.close()

if __name__ == "__main__":
    main()
