import torch
import cv2
import numpy as np
from matplotlib import pyplot as plt
from torch import nn
from torchvision import transforms as T
from PIL import Image
from ultralytics.utils.loss import v8DetectionLoss

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class YOLOv8WithClassification(nn.Module):
    def __init__(self, yolo_model, num_classes, class_id_to_name, detection_class_names):
        super(YOLOv8WithClassification, self).__init__()
        self.model = yolo_model.model  # YOLOv8 的底层 nn.Module
        self.num_classes = num_classes
        self.class_id_to_name = class_id_to_name  # 分类类别名称映射
        self.detection_class_names = detection_class_names  # 检测类别名称列表
        self.classification_head = None  # 分类头将在第一次前向传播时初始化
        self.checkbox_head = None  # 新增checkbox head
        self.features = None  # 用于存储中间特征

        if hasattr(self.model, 'args') and isinstance(self.model.args, dict):
            self.model.args = AttrDict(self.model.args)
        else:
            print("模型缺少 'args' 属性或 'args' 不是一个字典。手动定义 'args'。")
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
            raise ValueError("未能捕获特征，请检查前向钩子的设置。")

        gap = torch.mean(features, dim=(2, 3))
        if self.classification_head is None:
            feature_dim = gap.shape[1]
            self.classification_head = nn.Linear(feature_dim, self.num_classes).to(x.device)
        classification_logits = self.classification_head(gap)

        if self.checkbox_head is None:
            feature_dim = gap.shape[1]
            self.checkbox_head = nn.Linear(feature_dim, 1).to(x.device)
        checkbox_logits = self.checkbox_head(gap).squeeze(1)

        if targets is not None:
            detection_loss, _ = self.loss_func(predictions, {
                'batch_idx': targets['batch_idx'],
                'cls': targets['cls'],
                'bboxes': targets['bboxes']
            })
            checkbox_targets = targets['checkboxes']
            checkbox_loss = nn.BCEWithLogitsLoss()(checkbox_logits, checkbox_targets)
            total_loss = detection_loss + checkbox_loss
            return classification_logits, detection_loss, checkbox_loss, total_loss
        else:
            return classification_logits, checkbox_logits, predictions

# 加载模型
model_path = r'D:\Programming\Project\github\KonColle\KC\Models\yolov8_KC_model.pt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载模型权重
checkpoint = torch.load(model_path, map_location=device)

# 从 checkpoint 中直接获取模型，'model' 键已经包含了整个模型实例
model = checkpoint['model']

# 提取其他信息
class_id_to_name = checkpoint['class_id_to_name']
detection_class_names = checkpoint['detection_class_names']

# 重新创建 YOLOv8WithClassification 实例，传入必要参数
model = YOLOv8WithClassification(model, num_classes=len(class_id_to_name),
                                 class_id_to_name=class_id_to_name,
                                 detection_class_names=detection_class_names)

# 如果你还需要加载 'checkbox_head'，也可以从 checkpoint 中提取
model.checkbox_head = checkpoint['checkbox_head']

# 将模型移动到设备上
model = model.to(device)
model.eval()


# 图像预处理
transform = T.Compose([
    T.Resize((1120, 672)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 验证函数
def validate_image(image_path):
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    # 推理
    with torch.no_grad():
        classification_logits, checkbox_logits, predictions = model(image_tensor)

    # predictions 是一个包含边界框的张量，形状通常是 [num_boxes, 6]，
    # 其中前 4 列是边界框坐标，接下来是置信度和分类概率
    # 例如：[x1, y1, x2, y2, confidence, class_id]
    detection_results = predictions[0]  # 这里假设 predictions[0] 是一个 [num_boxes, 6] 的张量

    # 获取检测框和类别
    boxes = detection_results[:, :4]  # 前 4 列是边界框
    confidences = detection_results[:, 4]  # 第 5 列是置信度
    class_ids = detection_results[:, 5].long()  # 第 6 列是分类标签

    # 使用 torch.max 获取每个检测框的分类标签
    classification_labels = torch.argmax(classification_logits, dim=1)

    # 显示检测结果
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image)

    for i, box in enumerate(boxes):
        # 检查 box 的形状，确保它包含 4 个元素：x1, y1, x2, y2
        if len(box.shape) == 1 and box.shape[0] == 4:
            x1, y1, x2, y2 = box.cpu().numpy()
        else:
            print(f"Unexpected box shape: {box.shape}")
            continue  # 跳过不符合预期的框

        # 使用 .item() 确保坐标是标量
        x1, y1, x2, y2 = x1.item(), y1.item(), x2.item(), y2.item()

        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color='red', linewidth=2))
        label = detection_class_names[class_ids[i].item()]
        ax.text(x1, y1, label, color='yellow', fontsize=12, bbox=dict(facecolor='black', alpha=0.5))

    # 输出分类结果
    classification_label = class_id_to_name[classification_labels.item()]
    print(f"分类结果: {classification_label}")

    plt.show()



# 输入图片路径
image_path = r'D:\Programming\Project\github\KonColle\Datasets\images\waters\in_map\combat_result\combat_result_4.png'
validate_image(image_path)



# D:\Programming\Project\github\KonColle\Datasets\images\waters\map_1\map_1_1.png'