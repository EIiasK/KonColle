import cv2
import numpy as np
import mss
import torch
import time
from torchvision.ops import nms
from ultralytics import YOLO
from Model_Trainer_YOLOv8 import AttrDict, YOLOv8WithClassification

# 注册安全全局变量，确保 pickle 正确加载 YOLOv8WithClassification
torch.serialization.add_safe_globals({'YOLOv8WithClassification': YOLOv8WithClassification})

def process_raw_preds(raw_preds, device, conf_thresh):
    """
    统一处理 raw_preds，返回 boxes, obj_conf, class_conf 三个 tensor。
    处理流程：
      1. 如果 raw_preds 是 tuple：
           - 当 tuple 长度 < 3 时，取第一个元素作为预测张量
           - 当 tuple 长度 >= 3 时，取前3个元素分别为 boxes, obj_conf, class_conf
      2. 如果 raw_preds 是 tensor，则直接使用
      3. 对预测张量进行 squeeze 操作，并检查 shape 是否符合预期（列数>=6）
    """
    if isinstance(raw_preds, tuple):
        if len(raw_preds) < 3:
            pred_tensor = raw_preds[0]
            pred_tensor = pred_tensor.cpu()
            if pred_tensor.dim() == 3:
                pred_tensor = pred_tensor.squeeze(0)
            if pred_tensor.shape[1] < 6:
                return None, None, None
            boxes = pred_tensor[:, :4]
            obj_conf = pred_tensor[:, 4]
            class_conf = pred_tensor[:, 5:]
        else:
            # 假定前三个元素分别为 boxes, obj_conf, class_conf
            boxes, obj_conf, class_conf = raw_preds[:3]
            boxes = boxes.cpu()
            obj_conf = obj_conf.cpu()
            class_conf = class_conf.cpu()
            # 若 boxes 为 3d 张量，则尝试 squeeze 或 transpose（根据实际情况调整）
            if boxes.dim() == 3:
                # 如果第一维为1，则 squeeze；否则尝试转置
                if boxes.shape[0] == 1:
                    boxes = boxes.squeeze(0)
                elif boxes.shape[1] == 4:
                    boxes = boxes.permute(1, 0)
    else:
        pred_tensor = raw_preds.cpu()
        if pred_tensor.dim() == 3:
            pred_tensor = pred_tensor.squeeze(0)
        if pred_tensor.shape[1] < 6:
            return None, None, None
        boxes = pred_tensor[:, :4]
        obj_conf = pred_tensor[:, 4]
        class_conf = pred_tensor[:, 5:]
    return boxes, obj_conf, class_conf

def main():
    # 修改为你的实际模型保存路径
    model_path = r'D:\Programming\Project\github\KonColle\KC\Models\yolov8_KC_model.pt'

    # 从 checkpoint 中读取类别信息
    checkpoint = torch.load(model_path, map_location='cpu')
    detection_class_names = checkpoint.get('detection_class_names', None)
    classification_class_id_to_name = checkpoint.get('class_id_to_name', None)

    # 直接加载保存的自定义模型（YOLOv8WithClassification实例）
    model = checkpoint['model']
    model.eval()  # 切换到推理模式
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 设置检测类别名称（仅用于显示标签）
    if detection_class_names is not None:
        if isinstance(detection_class_names, dict):
            detection_names_list = [detection_class_names[k] for k in sorted(detection_class_names.keys())]
        else:
            detection_names_list = detection_class_names
        if hasattr(model, 'detection_class_names'):
            model.detection_class_names = detection_names_list
        if hasattr(model, 'model') and hasattr(model.model, 'yaml'):
            model.model.yaml['names'] = detection_names_list
    else:
        print("未找到检测类别名称，无法设置类别信息。")

    # 输出类别信息
    if detection_class_names:
        print("检测模型可以识别的所有类别及其标签：")
        if isinstance(detection_class_names, dict):
            for class_id, class_name in detection_class_names.items():
                print(f"检测类别 {class_id}: {class_name}")
        elif isinstance(detection_class_names, list):
            for idx, name in enumerate(detection_class_names):
                print(f"检测类别 {idx}: {name}")
    else:
        print("未找到检测类别名称。")
    if classification_class_id_to_name:
        print("\n分类模型可以识别的所有类别及其标签：")
        for class_id, class_name in classification_class_id_to_name.items():
            print(f"分类类别 {class_id}: {class_name}")
    else:
        print("未找到分类类别名称。")

    # 实时屏幕检测：使用 mss 捕获整个屏幕
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # 主屏幕
        last_print_time = time.time()
        conf_thresh = 0.25  # 置信度阈值
        iou_thresh = 0.45   # NMS IoU 阈值

        while True:
            sct_img = sct.grab(monitor)
            frame = np.array(sct_img)
            # 转换 BGRA 至 BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            orig_h, orig_w = frame.shape[:2]

            # 调整图像尺寸（与训练时保持一致，例如1120x672）
            target_w, target_h = 1120, 672
            frame_resized = cv2.resize(frame, (target_w, target_h))
            img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img_rgb).float() / 255.0  # 归一化
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)    # 变为 1x3xHxW
            # 标准化（使用训练时的均值和标准差）
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            img_tensor = (img_tensor - mean) / std
            img_tensor = img_tensor.to(device)

            # 模型推理
            with torch.no_grad():
                class_logits, checkbox_logits, raw_preds = model(img_tensor)

            # 处理检测输出，统一得到 boxes, obj_conf, class_conf
            boxes, obj_conf, class_conf = process_raw_preds(raw_preds, device, conf_thresh)
            if boxes is None or obj_conf is None or class_conf is None:
                print("检测预测输出格式异常")
                continue

            # 计算每个预测的类别得分及最终得分（目标置信度 * 类别得分）
            class_scores, class_ids = torch.max(class_conf, dim=1)
            scores = obj_conf * class_scores

            # 过滤掉低于阈值的预测
            mask = scores >= conf_thresh
            boxes = boxes[mask]
            scores = scores[mask]
            class_ids = class_ids[mask]

            # 对每个类别进行非极大值抑制（NMS）
            final_indices = []
            if boxes.shape[0] > 0:
                for cls in class_ids.unique():
                    cls_mask = class_ids == cls
                    cls_boxes = boxes[cls_mask]
                    cls_scores = scores[cls_mask]
                    keep = nms(cls_boxes, cls_scores, iou_thresh)
                    cls_indices = torch.nonzero(cls_mask, as_tuple=False).squeeze(1)
                    final_indices.append(cls_indices[keep])
                if final_indices:
                    final_indices = torch.cat(final_indices).unique()
                else:
                    final_indices = torch.tensor([], dtype=torch.long)
            else:
                final_indices = torch.tensor([], dtype=torch.long)

            # 绘制检测框和标签
            annotated_frame = frame.copy()
            for idx in final_indices.tolist():
                # 将检测框坐标缩放回原始屏幕尺寸
                x1, y1, x2, y2 = boxes[idx]
                x1 = int(x1.item() * (orig_w / target_w))
                y1 = int(y1.item() * (orig_h / target_h))
                x2 = int(x2.item() * (orig_w / target_w))
                y2 = int(y2.item() * (orig_h / target_h))
                cls_id = int(class_ids[idx].item())
                score_val = float(scores[idx].item())
                # 获取检测类别名称
                if detection_class_names is not None:
                    if isinstance(detection_class_names, dict):
                        cls_name = detection_class_names.get(cls_id, str(cls_id))
                    else:
                        cls_name = detection_class_names[cls_id] if cls_id < len(detection_class_names) else str(cls_id)
                else:
                    cls_name = str(cls_id)
                # 固定颜色（利用类别ID种子保证同类颜色一致）
                color = tuple(int(c) for c in np.random.RandomState(cls_id).randint(0, 255, size=3))
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                label = f"{cls_name} {score_val:.2f}"
                (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                tx1, ty1 = x1, y1 - th - 3
                tx2, ty2 = x1 + tw + 2, y1
                if ty1 < 0:
                    ty1 = y1
                    ty2 = y1 + th + 3
                cv2.rectangle(annotated_frame, (tx1, ty1), (tx2, ty2), color, -1)
                text_color = (255, 255, 255) if (color[0]*0.299 + color[1]*0.587 + color[2]*0.114) < 186 else (0, 0, 0)
                cv2.putText(annotated_frame, label, (x1+1, ty1+th), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)

            cv2.imshow("Screen Detection", annotated_frame)

            # 每隔2秒输出一次分类结果到终端
            if classification_class_id_to_name and time.time() - last_print_time >= 2:
                probs = torch.softmax(class_logits, dim=1)[0].cpu().numpy()
                cls_idx = int(np.argmax(probs))
                cls_prob = probs[cls_idx]
                cls_name = classification_class_id_to_name.get(cls_idx, str(cls_idx))
                print(f"当前屏幕分类: {cls_name} ({cls_prob:.2f})")
                last_print_time = time.time()

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
