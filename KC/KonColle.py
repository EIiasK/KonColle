import cv2
import numpy as np
import torch
from ultralytics import YOLO

# 定义 AttrDict 以支持反序列化训练时保存的 checkpoint
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

# 注册到 safe_globals 提高安全性
try:
    torch.serialization.add_safe_globals({'AttrDict': AttrDict})
except Exception:
    pass


def main():
    # 1. 配置路径和设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = r"D:\Programming\Project\github\KonColle\KC\Models\KC_Detection_model.pt"
    base_model_path = r"D:\Programming\Project\github\KonColle\KC\Models\YOLOv8\yolov8n.pt"

    # 2. 加载 checkpoint 自定义类别
    checkpoint = torch.load(ckpt_path, map_location=device)
    det_names = checkpoint.get('detection_class_names', [])
    num_classes = len(det_names)
    print("Detection classes:", det_names)

    # 3. 初始化 YOLOv8 模型并加载权重
    model = YOLO(base_model_path)
    # 注入训练好的模型参数
    model.model.load_state_dict(checkpoint['model'].state_dict())
    # 更新底层分类数量
    model.model.yaml['nc'] = num_classes
    model.model.yaml['names'] = det_names
    model.to(device)
    model.model.eval()

    # 4. 读取并预处理静态图片
    img_path = "test.jpg"
    frame = cv2.imread(img_path)
    if frame is None:
        print(f"Failed to load image: {img_path}")
        return
    orig_h, orig_w = frame.shape[:2]
    # 使用 YOLO wrapper 自动预处理，但后面手动绘制，自行获取原始坐标
    results = model(frame, conf=0.01, imgsz=(672,1120), device=device)[0]

    # 5. 手动绘制检测框和自定义标签
    annotated = frame.copy()
    boxes = results.boxes.xyxy.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy().astype(int)
    for (x1, y1, x2, y2), conf, cls_id in zip(boxes, confs, classes):
        name = det_names[cls_id] if cls_id < len(det_names) else str(cls_id)
        label = f"{name} {conf:.2f}"
        # 绘制边框
        cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
        # 绘制标签背景
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(annotated, (int(x1), int(y1)-th-4), (int(x1)+tw, int(y1)), (0,255,0), -1)
        # 绘制标签文字
        cv2.putText(annotated, label, (int(x1), int(y1)-4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

    # 6. 显示结果
    cv2.imshow("Detection", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
