import torch
from matplotlib import pyplot as plt
from torch import nn
from torchvision import transforms as T
from PIL import Image
from ultralytics.utils.loss import v8DetectionLoss
from Model_Trainer_YOLOv8 import AttrDict, YOLOv8WithClassification


# 修改后的检验程序，直接加载训练保存的模型实例
model_path = r'D:\Programming\Project\github\KonColle\KC\Models\yolov8_KC_model.pt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 直接加载整个模型实例，训练时已保存了完整的模型（含分类头、checkbox head及类别信息）
checkpoint = torch.load(model_path, map_location=device)
model = checkpoint['model']
model.eval()
model = model.to(device)

# 使用训练阶段的类别映射信息
class_id_to_name = model.class_id_to_name
detection_class_names = model.detection_class_names

# 图像预处理：与训练阶段保持一致的尺寸和归一化参数
transform = T.Compose([
    T.Resize((672, 1120)),  # 与训练阶段一致：(img_height, img_width) = (672, 1120)
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def validate_image(image_path):
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    # 推理
    with torch.no_grad():
        classification_logits, checkbox_logits, predictions = model(image_tensor)

    # 假设 predictions[0] 为 [num_boxes, 6] 张量，其中前 4 列为边界框
    detection_results = predictions[0]
    boxes = detection_results[:, :4]
    confidences = detection_results[:, 4]
    class_ids = detection_results[:, 5].long()

    # 计算分类结果
    classification_labels = torch.argmax(classification_logits, dim=1)

    # 显示检测结果
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image)

    for i, box in enumerate(boxes):
        if len(box.shape) == 1 and box.shape[0] == 4:
            x1, y1, x2, y2 = box.cpu().numpy()
        else:
            print(f"Unexpected box shape: {box.shape}")
            continue
        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color='red', linewidth=2))
        label = detection_class_names[class_ids[i].item()]
        ax.text(x1, y1, label, color='yellow', fontsize=12, bbox=dict(facecolor='black', alpha=0.5))

    # 输出分类结果
    classification_label = class_id_to_name[classification_labels.item()]
    print(f"分类结果: {classification_label}")
    plt.show()

# 输入图片路径进行检验
image_path = r'D:\Programming\Project\github\KonColle\Datasets\images\waters\map_1\map_1_1.png'
validate_image(image_path)



# D:\Programming\Project\github\KonColle\Datasets\images\waters\map_1\map_1_1.png'