import cv2
import numpy as np
import mss
import torch
from Model_Trainer_YOLOv8 import YOLOv8WithClassification, AttrDict
import torchvision.transforms as T
import time

# 注册自定义模型类，确保反序列化时可用
torch.serialization.add_safe_globals({'YOLOv8WithClassification': YOLOv8WithClassification})

def main():
    # 1. 加载训练好的模型
    custom_model_path = r"D:\Programming\Project\github\KonColle\KC\Models\yolov8_KC_model.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(custom_model_path, map_location=device)
    custom_model = checkpoint['model']
    custom_model.to(device)
    custom_model.eval()

    # 提取检测类别名称，并更新到模型内部配置
    detection_class_names = checkpoint.get("detection_class_names", [])
    custom_model.detection_class_names = detection_class_names
    custom_model.model.yaml['names'] = detection_class_names
    print("模型检测类别：", detection_class_names)

    # 提取分类映射，用于显示分类结果
    class_id_to_name = checkpoint.get("class_id_to_name", None)
    if class_id_to_name is None:
        class_id_to_name = {i: name for i, name in enumerate(detection_class_names)}
    print("模型分类映射：", class_id_to_name)

    # 2. 设置输入尺寸和归一化预处理（与训练时保持一致）
    target_w, target_h = 1120, 672
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # 3. 推理循环
    last_cls_print = time.time()
    frame_count = 0
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # 主屏幕
        while True:
            frame_count += 1
            # 捕获屏幕并转换格式
            sct_img = sct.grab(monitor)
            frame = np.array(sct_img)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            orig_h, orig_w = frame.shape[:2]

            # 预处理：调整尺寸、归一化
            frame_resized = cv2.resize(frame, (target_w, target_h))
            img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(device, dtype=torch.float32) / 255.0
            img_tensor = normalize(img_tensor)

            # 分类推理
            with torch.no_grad():
                cls_logits, _ = custom_model(img_tensor)
            cls_probs = torch.softmax(cls_logits, dim=1)[0]
            cls_idx = int(torch.argmax(cls_probs))
            cls_conf = float(cls_probs[cls_idx])
            cls_name = class_id_to_name.get(cls_idx, str(cls_idx))
            if time.time() - last_cls_print >= 2:
                print(f"当前屏幕分类: {cls_name} ({cls_conf:.2f})")
                last_cls_print = time.time()

            # 目标检测推理
            results = custom_model.model.predict(img_tensor)
            res = results[0]
            # 获取检测框、置信度和类别
            if hasattr(res, 'boxes') and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy()    # (n,4)
                confs = res.boxes.conf.cpu().numpy()   # (n,)
                classes = res.boxes.cls.cpu().numpy().astype(int)  # (n,)
            else:
                xyxy = np.empty((0, 4))
                confs = np.array([])
                classes = np.array([])

            # 可视化
            annotated_frame = frame.copy()
            scale_x = orig_w / target_w
            scale_y = orig_h / target_h
            for (x1r, y1r, x2r, y2r), conf, cls_id in zip(xyxy, confs, classes):
                x1 = int(x1r * scale_x)
                y1 = int(y1r * scale_y)
                x2 = int(x2r * scale_x)
                y2 = int(y2r * scale_y)
                det_cls_name = detection_class_names[cls_id] if cls_id < len(detection_class_names) else str(cls_id)
                # 颜色固定
                rand = np.random.RandomState(cls_id)
                color = tuple(int(c) for c in rand.randint(0, 255, size=3))
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                label = f"{det_cls_name} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                ty = y1 - th - 3 if y1 - th - 3 > 0 else y1
                cv2.rectangle(annotated_frame, (x1, ty), (x1 + tw + 2, ty + th), color, -1)
                text_color = (255, 255, 255) if (color[0]*0.299 + color[1]*0.587 + color[2]*0.114) < 186 else (0, 0, 0)
                cv2.putText(annotated_frame, label, (x1 + 1, ty + th - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)

            cv2.imshow("Screen Detection", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
