import cv2
import numpy as np
import time
from PIL import ImageGrab
import torch
from ultralytics import YOLO

# 定义 AttrDict ，支持加载训练时保存的 checkpoint
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

# 注册 AttrDict 到 safe_globals 增强安全性
try:
    torch.serialization.add_safe_globals({'AttrDict': AttrDict})
except Exception:
    pass


def main():
    # 1. 配置设备与模型路径
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = r"D:\Programming\Project\github\KonColle\KC\Models\KC_Detection_model.pt"
    base_model_path = r"D:\Programming\Project\github\KonColle\KC\Models\YOLOv8\yolov8n.pt"

    # 2. 加载 checkpoint 并获取自定义类别
    checkpoint = torch.load(ckpt_path, map_location=device)
    det_names = checkpoint.get('detection_class_names', [])
    print("检测类别：", det_names)

    # 3. 初始化 YOLOv8 模型并注入权重与类别
    model = YOLO(base_model_path)
    model.model.load_state_dict(checkpoint['model'].state_dict())
    model.model.yaml['nc'] = len(det_names)
    model.model.yaml['names'] = det_names
    try:
        model.names = {i: name for i, name in enumerate(det_names)}
    except Exception:
        pass
    model.to(device)
    model.model.eval()

    # 4. ROI 坐标（全屏分辨率）
    X, Y, W, H = 0, 312, 1512, 952

    # 5. 创建与 ROI 同尺寸的显示窗口（无需缩放）
    window_name = "实时目标检测"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, W, H)

    # 6. 循环抓屏并检测，帧率 ~5 FPS
    target_fps = 5
    frame_interval = 1.0 / target_fps
    while True:
        start_time = time.time()

        # 6.1 捕获全屏并裁剪 ROI 区域
        screenshot = ImageGrab.grab()
        full = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        crop = full[Y:Y+H, X:X+W]

        # 6.2 模型推理，仅做一次缩放到 (1120,672)
        results = model(crop, conf=0.005, imgsz=(1120, 672), device=device)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)

        # 6.3 在 crop 上绘制检测结果
        annotated = crop.copy()
        for (x1, y1, x2, y2), conf, cls_id in zip(boxes, confs, classes):
            name = det_names[cls_id] if cls_id < len(det_names) else str(cls_id)
            label = f"{name} {conf:.2f}"
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(annotated, (int(x1), int(y1) - th - 4), (int(x1) + tw, int(y1)), (0, 255, 0), -1)
            cv2.putText(annotated, label, (int(x1), int(y1) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # 6.4 直接显示 annotated（与 ROI 同尺寸）
        cv2.imshow(window_name, annotated)
        cv2.waitKey(1)  # 保持窗口响应

        # 6.5 控制帧率同时处理事件
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break
        elapsed = time.time() - start_time
        delay = frame_interval - elapsed
        if delay > 0:
            end_t = time.time() + delay
            while time.time() < end_t:
                cv2.waitKey(1)
                time.sleep(0.01)

    # 7. 清理并关闭窗口
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
