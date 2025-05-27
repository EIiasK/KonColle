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
from collections import OrderedDict
from ultralytics import YOLO
from ultralytics.utils.loss import v8DetectionLoss

# 抑制冗长日志输出
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# 支持属性访问的字典类
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

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

# 提取COCO标注文件中的检测类别名称，返回类别ID到名称的映射
def extract_detection_classes(annotation_file):
    with open(annotation_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    categories = sorted(data.get('categories', []), key=lambda x: x['id'])
    detection_class_id_to_name = OrderedDict()
    for idx, category in enumerate(categories):
        class_id = idx  # 赋予连续的0-based类别ID
        class_name = category['name']
        detection_class_id_to_name[class_id] = class_name
    return detection_class_id_to_name

def get_category_image_paths(base_dir):
    category_image_paths = {}
    for root, dirs, files in os.walk(base_dir):
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        relative_path = os.path.relpath(root, base_dir)
        category = os.path.basename(relative_path)
        if image_files:
            image_paths = [os.path.join(relative_path, f) for f in image_files]
            category_image_paths[category] = image_paths
            print(f"类别 '{category}' 的图片数量：{len(image_paths)}")
        else:
            print(f"类别 '{category}' 中没有找到图片文件。")
    return category_image_paths

def split_dataset(category_image_paths, test_size=0.2, random_state=42):
    train_images = []
    val_images = []
    for category, image_paths in category_image_paths.items():
        if not image_paths:
            print(f"类别 '{category}' 中没有图片可供划分。")
            continue
        train_imgs, val_imgs = train_test_split(image_paths, test_size=test_size, random_state=random_state)
        train_images.extend(train_imgs)
        val_images.extend(val_imgs)
    return train_images, val_images

def filter_coco_annotations(annotation_file, image_files, output_file):
    with open(annotation_file, 'r') as f:
        coco_data = json.load(f)
    # 根据图像文件列表过滤COCO标注
    filename_to_id = {img['file_name']: img['id'] for img in coco_data['images']}
    selected_image_ids = set()
    selected_images = []
    for img in coco_data['images']:
        if img['file_name'] in image_files:
            selected_images.append(img)
            selected_image_ids.add(img['id'])
    selected_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] in selected_image_ids]
    # 构建新的COCO标注数据并保存
    new_coco_data = {
        'info': coco_data.get('info', {}),
        'licenses': coco_data.get('licenses', []),
        'images': selected_images,
        'annotations': selected_annotations,
        'categories': coco_data.get('categories', [])
    }
    with open(output_file, 'w') as f:
        json.dump(new_coco_data, f)

# ==================== 自定义数据集类 ====================

# …（前面的导入和辅助函数保持不变，不再包含与 checkbox 相关的逻辑）

class CustomCocoDataset(CocoDetection):
    def __init__(self, root, annotation_file, transform=None, image_files=None, class_name_to_id=None):
        super().__init__(root=root, annFile=annotation_file)
        self.transform = transform
        if class_name_to_id is None:
            raise ValueError("必须提供 class_name_to_id 映射。")
        self.class_name_to_id = class_name_to_id  # 分类类别名称到ID映射
        self.class_id_to_name = {idx: name for name, idx in class_name_to_id.items()}
        if image_files is not None:
            filename_to_id = {img_info['file_name']: img_id for img_id, img_info in self.coco.imgs.items()}
            self.ids = [filename_to_id[fname] for fname in image_files if fname in filename_to_id]
        else:
            self.ids = list(self.coco.imgs.keys())

    def __getitem__(self, idx):
        img, ann = super().__getitem__(idx)
        if self.transform is not None:
            img = self.transform(img)
        img_info = self.coco.imgs[self.ids[idx]]
        orig_width = img_info['width']
        orig_height = img_info['height']
        boxes = []
        labels = []
        area = []
        iscrowd = []
        for obj in ann:
            xmin, ymin, width, height = obj['bbox']
            x_center = (xmin + width / 2.0) / orig_width
            y_center = (ymin + height / 2.0) / orig_height
            w_norm = width / orig_width
            h_norm = height / orig_height
            boxes.append([x_center, y_center, w_norm, h_norm])
            labels.append(obj['category_id'] - 1)  # 转换为0-based类别ID
            area.append(obj['area'])
            iscrowd.append(obj.get('iscrowd', 0))
        target = {
            'bboxes': torch.as_tensor(boxes, dtype=torch.float32),
            'cls': torch.as_tensor(labels, dtype=torch.int64),
            'image_id': torch.tensor([self.ids[idx]]),
            'area': torch.as_tensor(area, dtype=torch.float32),
            'iscrowd': torch.as_tensor(iscrowd, dtype=torch.int64)
        }
        img_file_name = img_info['file_name']
        category_name = os.path.basename(os.path.dirname(img_file_name))
        if category_name not in self.class_name_to_id:
            raise KeyError(f"类别名称 '{category_name}' 未在 class_name_to_id 中找到!")
        classification_label = torch.tensor(self.class_name_to_id[category_name], dtype=torch.long)
        return img, target, classification_label

# 带分类头的自定义YOLOv8模型（已移除 checkbox 部分）
class YOLOv8WithClassification(nn.Module):
    def __init__(self, yolo_model, num_classes, class_id_to_name, detection_class_names):
        super(YOLOv8WithClassification, self).__init__()
        self.model = yolo_model.model  # YOLOv8 底层模型 (nn.Module)
        self.num_classes = num_classes
        self.class_id_to_name = class_id_to_name
        self.detection_class_names = detection_class_names
        self.classification_head = None  # 分类头（延迟初始化）
        self.features = None  # 中间特征存储

        if hasattr(self.model, 'args') and isinstance(self.model.args, dict):
            self.model.args = AttrDict(self.model.args)
        else:
            print("模型缺少 'args' 属性或 'args' 不是字典，使用默认超参数。")
            self.model.args = AttrDict({
                'box': 7.5,
                'cls': 0.5,
                'obj': 1.0,
                'iou': 0.20,
                'lr0': 0.01,
                'lrf': 0.01,
            })
        if not hasattr(self.model.args, 'box'):
            self.model.args.box = 7.5

        self._register_hook()
        self.loss_func = v8DetectionLoss(self.model)

    def hook(self, module, input, output):
        self.features = output

    def _register_hook(self):
        if len(self.model.model) >= 2:
            self.model.model[-2].register_forward_hook(self.hook)
        else:
            print("模型结构不符合预期，无法注册钩子。")

    def forward(self, x, targets=None):
        self.features = None
        predictions = self.model(x)
        features = self.features
        if features is None:
            raise ValueError("未捕获中间特征，请检查钩子设置。")
        gap = torch.mean(features, dim=(2, 3))
        if self.classification_head is None:
            feature_dim = gap.shape[1]
            self.classification_head = nn.Linear(feature_dim, self.num_classes).to(x.device)
        classification_logits = self.classification_head(gap)
        if targets is not None:
            detection_loss, _ = self.loss_func(predictions, {
                'batch_idx': targets['batch_idx'],
                'cls': targets['cls'],
                'bboxes': targets['bboxes']
            })
            total_loss = detection_loss
            return classification_logits, detection_loss, total_loss
        else:
            return classification_logits, predictions

def main():
    # 参数配置
    base_dir = r"D:\Programming\Project\github\KonColle\Datasets\images"
    annotation_file = r"D:\Programming\Project\github\KonColle\Datasets\annotations\KonColle_coco_fixed.json"
    img_width, img_height = 1120, 672
    batch_size = 8
    epochs = 30
    learning_rate = 1e-4
    model_save_path = r"D:\Programming\Project\github\KonColle\KC\Models\yolov8_KC_model.pt"
    log_dir = r"D:\Programming\Project\github\KonColle\KC\Logs"

    writer = SummaryWriter(log_dir=log_dir)

    # 数据预处理
    train_transforms = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transforms = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    category_image_paths = get_category_image_paths(base_dir)
    if not category_image_paths:
        print("未找到任何类别的图片文件，请检查数据集路径。")
        return

    train_images, val_images = split_dataset(category_image_paths, test_size=0.2, random_state=42)
    train_annotation_file = r"D:\Programming\Project\github\KonColle\Datasets\annotations\instances_train.json"
    val_annotation_file = r"D:\Programming\Project\github\KonColle\Datasets\annotations\instances_val.json"
    filter_coco_annotations(annotation_file, train_images, train_annotation_file)
    filter_coco_annotations(annotation_file, val_images, val_annotation_file)

    detection_class_id_to_name = extract_detection_classes(train_annotation_file)
    num_detection_classes = len(detection_class_id_to_name)
    detection_class_names = list(detection_class_id_to_name.values())
    print(f"提取的检测类别数量: {num_detection_classes}")
    print(f"detection_class_names: {detection_class_names}")
    classification_class_names = sorted(category_image_paths.keys())
    classification_class_name_to_id = {name: idx for idx, name in enumerate(classification_class_names)}
    classification_class_id_to_name = {idx: name for name, idx in classification_class_name_to_id.items()}
    print(f"分类类别名称到ID的映射: {classification_class_name_to_id}")

    train_dataset = CustomCocoDataset(
        root=base_dir,
        annotation_file=train_annotation_file,
        transform=train_transforms,
        image_files=train_images,
        class_name_to_id=classification_class_name_to_id
    )
    val_dataset = CustomCocoDataset(
        root=base_dir,
        annotation_file=val_annotation_file,
        transform=val_transforms,
        image_files=val_images,
        class_name_to_id=classification_class_name_to_id
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=custom_collate_fn)

    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print("训练集类别分布:", get_class_distribution(train_dataset))
    print("验证集类别分布:", get_class_distribution(val_dataset))

    # 构建模型
    yolov8_model_path = r"D:\Programming\Project\github\KonColle\KC\Models\YOLOv8\yolov8n.pt"
    yolov8_model = YOLO(yolov8_model_path)
    yolov8_model.model.yaml['nc'] = num_detection_classes
    yolov8_model.model.yaml['names'] = detection_class_names

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    yolov8_model.model.to(device)

    default_hyp = {
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'pose': 12.0,
        'kobj': 1.0,
        'overlap_mask': True,
        'mask_ratio': 4.0,
    }
    if not hasattr(yolov8_model.model, 'args') or yolov8_model.model.args is None:
        yolov8_model.model.args = {}
    yolov8_model.model.args.update(default_hyp)
    if not isinstance(yolov8_model.model.args, AttrDict):
        yolov8_model.model.args = AttrDict(yolov8_model.model.args)

    num_classes = len(classification_class_name_to_id)
    model = YOLOv8WithClassification(
        yolo_model=yolov8_model,
        num_classes=num_classes,
        class_id_to_name=classification_class_id_to_name,
        detection_class_names=detection_class_names
    )
    for param in model.parameters():
        param.requires_grad = True

    print(f"类别数量 (分类): {num_classes}")
    print(f"分类类别ID->名称: {train_dataset.class_id_to_name}")
    print(f"检测类别名称列表: {model.detection_class_names}")
    print("模型加载并配置完成")

    classification_criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    scaler = GradScaler()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float("inf")

    # 训练和验证循环
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
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

            for images, detection_targets, classification_targets in dataloader:
                # 将图像堆叠为一个Tensor并移动到设备
                images = torch.stack([img.to(device) for img in images])
                classification_targets = classification_targets.to(device)

                # 构建 batch 级别的 detection targets
                target_list = []
                for i, target in enumerate(detection_targets):
                    if 'cls' not in target:
                        print(f"警告: 第 {i} 个目标缺少 'cls' 键")
                        continue
                    boxes = target['bboxes'].to(device)  # [n,4]
                    labels = target['cls'].to(device).unsqueeze(1)
                    batch_idx_tensor = torch.full((labels.size(0), 1), i, dtype=torch.long, device=device)
                    targets_per_image = torch.cat([batch_idx_tensor, labels, boxes], dim=1)
                    target_list.append(targets_per_image)
                if target_list:
                    targets = torch.cat(target_list, dim=0)
                    batch_targets = {
                        'batch_idx': targets[:, 0].long(),
                        'cls': targets[:, 1].long(),
                        'bboxes': targets[:, 2:6],
                    }
                else:
                    batch_targets = {
                        'batch_idx': torch.tensor([], dtype=torch.long, device=device),
                        'cls': torch.tensor([], dtype=torch.long, device=device),
                        'bboxes': torch.tensor([], dtype=torch.float32, device=device),
                    }

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    with torch.autocast(device.type, dtype=torch.float16):
                        classification_logits, detection_loss, total_loss = model(images, batch_targets)
                        classification_loss = classification_criterion(classification_logits, classification_targets)
                        total_loss += classification_loss  # 综合总损失
                    if phase == "train":
                        scaler.scale(total_loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                batch_size_current = images.size(0)
                running_loss += total_loss.item() * batch_size_current
                running_classification_loss += classification_loss.item() * batch_size_current
                running_detection_loss += detection_loss.item() * batch_size_current
                _, preds = torch.max(classification_logits, 1)
                running_corrects += torch.sum(preds == classification_targets.data)
                running_total += batch_size_current

            epoch_loss = running_loss / running_total if running_total > 0 else 0.0
            epoch_classification_loss = running_classification_loss / running_total if running_total > 0 else 0.0
            epoch_detection_loss = running_detection_loss / running_total if running_total > 0 else 0.0
            epoch_acc = (running_corrects.double() / running_total) if running_total > 0 else 0.0

            print(f"{phase} Loss: {epoch_loss:.4f} (分类: {epoch_classification_loss:.4f}, 检测: {epoch_detection_loss:.4f}) Acc: {epoch_acc:.4f}")

            if phase == "val":
                scheduler.step(epoch_loss)
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save({
                        'model': model,
                        'class_id_to_name': model.class_id_to_name,
                        'detection_class_names': model.detection_class_names
                    }, model_save_path.replace('.pth', '.pt'))
                    print(f"最佳模型已保存到: {model_save_path.replace('.pth', '.pt')}")
                    print("保存的检测类别名称:", model.detection_class_names)
                    print("保存的分类类别名称:", model.class_id_to_name)
        print("-" * 10)
    print("训练完成")
    model.load_state_dict(best_model_wts)
    torch.save({
        'model': model,
        'class_id_to_name': model.class_id_to_name,
        'detection_class_names': model.detection_class_names
    }, model_save_path.replace('.pth', '.pt'))
    print(f"最终模型已保存到: {model_save_path.replace('.pth', '.pt')}")
    writer.close()

if __name__ == "__main__":
    main()
