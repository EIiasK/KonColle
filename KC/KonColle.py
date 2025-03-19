import cv2
import numpy as np
import mss
import torch
import time
from ultralytics import YOLO

# 如果需要注册自定义模型类以便反序列化（必须保证训练时保存的 checkpoint 能正常加载）
from Model_Trainer_YOLOv8 import YOLOv8WithClassification, AttrDict

# 注册自定义模型类，确保反序列化时可用
torch.serialization.add_safe_globals({'YOLOv8WithClassification': YOLOv8WithClassification})


def main():
    # 模型权重路径（训练时保存的checkpoint，包含自定义分类头等）
    custom_model_path = r"D:\Programming\Project\github\KonColle\KC\Models\yolov8_KC_model.pt"
    # 同时用于检测的权重（Ultralytics官方API需要完整的 YOLO 对象）
    det_model_path = r"D:\Programming\Project\github\KonColle\KC\Models\YOLOv8\yolov8n.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------------
    # 加载检测模型：使用 Ultralytics YOLO API的 predict 接口
    det_model = YOLO(det_model_path)
    # 配置检测类别和类别名称
    # 这里假设检测类别由训练时提供的类别决定
    # 如果需要手动设置，可以修改 det_model.model.yaml['nc'] 和 ['names']
    # 本例中直接使用权重中保存的类别（也可以从 checkpoint 中提取）
    print("检测模型类别：", det_model.names)
    det_model.model.to(device)

    # ---------------------------
    # 加载自定义模型（含分类头）
    # 注意：训练时保存的 checkpoint中，"model"即为 YOLOv8WithClassification 对象
    checkpoint = torch.load(custom_model_path, map_location=device)
    custom_model = checkpoint['model']
    custom_model.eval()
    custom_model.to(device)
    print("分类模型类别映射：", custom_model.class_id_to_name)

    # 设置模型输入尺寸（需与训练时一致）
    target_w, target_h = 1120, 672

    # 使用 mss 获取屏幕截图
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # 主屏幕

        last_cls_print = time.time()

        while True:
            sct_img = sct.grab(monitor)
            frame = np.array(sct_img)
            # mss 返回 BGRA 格式，转换为 BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            orig_h, orig_w = frame.shape[:2]

            # 调整尺寸到模型输入尺寸
            frame_resized = cv2.resize(frame, (target_w, target_h))
            # 转为 RGB（Ultralytics 模型默认使用 RGB）
            img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

            # ---------------------------
            # 目标检测：调用 det_model.predict
            # det_model.predict 内部完成了预处理、前向和后处理（NMS 等）
            det_results = det_model.predict(img_rgb, conf=0.50, iou=0.45, verbose=False)
            det_result = det_results[0]  # 单帧预测结果

            # ---------------------------
            # 全屏分类：利用自定义模型的分类头
            # 构造张量（归一化处理同训练时一致）
            img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(device, dtype=torch.float32) / 255.0
            # 这里我们不做额外归一化，因为训练时使用了 torchvision.transforms.Normalize，
            # 你也可以加上 Normalize(mean, std)（这里以ImageNet均值/std为例）
            # 如： (img_tensor - mean) / std
            with torch.no_grad():
                classification_logits, _, _ = custom_model(img_tensor)
            # 分类结果：计算 softmax 得到概率分布
            cls_probs = torch.softmax(classification_logits, dim=1)
            cls_idx = int(torch.argmax(cls_probs[0]))
            cls_conf = float(cls_probs[0, cls_idx])
            cls_name = custom_model.class_id_to_name.get(cls_idx, str(cls_idx))

            # 每隔2秒在终端打印一次全屏分类结果
            if time.time() - last_cls_print >= 2:
                print(f"当前屏幕分类: {cls_name} ({cls_conf:.2f})")
                last_cls_print = time.time()

            # ---------------------------
            # 绘制检测结果
            annotated_frame = frame.copy()
            if det_result.boxes is not None:
                for box in det_result.boxes:
                    # 获取检测框坐标（xyxy格式），注意：返回的box.xyxy为 tensor，取第一个检测框的坐标
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    # 坐标基于 frame_resized 尺寸，还原到原始屏幕尺寸
                    x1 = int(x1 * (orig_w / target_w))
                    y1 = int(y1 * (orig_h / target_h))
                    x2 = int(x2 * (orig_w / target_w))
                    y2 = int(y2 * (orig_h / target_h))
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    cls_det = det_model.names[cls_id] if det_model.names and cls_id < len(det_model.names) else str(
                        cls_id)
                    # 固定同一类别的颜色
                    color = tuple(int(c) for c in np.random.RandomState(cls_id).randint(0, 255, size=3))
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    label = f"{cls_det} {conf:.2f}"
                    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    tx1, ty1 = x1, y1 - th - 3
                    if ty1 < 0:
                        ty1 = y1
                    cv2.rectangle(annotated_frame, (tx1, ty1), (tx1 + tw + 2, ty1 + th), color, -1)
                    text_color = (255, 255, 255) if (color[0] * 0.299 + color[1] * 0.587 + color[
                        2] * 0.114) < 186 else (0, 0, 0)
                    cv2.putText(annotated_frame, label, (x1 + 1, ty1 + th - 1),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)

            cv2.imshow("Screen Detection", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
