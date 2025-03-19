import cv2
import numpy as np
import mss
import torch
import time
from torchvision.ops import nms
from ultralytics import YOLO
from Model_Trainer_YOLOv8 import AttrDict, YOLOv8WithClassification

# 注册自定义模型类，确保反序列化时可用
torch.serialization.add_safe_globals({'YOLOv8WithClassification': YOLOv8WithClassification})


def process_raw_preds(raw_preds, target_w, target_h):
    """
    解析 raw_preds 并只截取前 18 个通道（假定输出格式为：
    [cx, cy, w, h, class_logits(共14)]）。

    如果前 4 个通道的最大值 <= 1，则认为是归一化值，否则认为是像素单位。
    最后将中心格式 (cx,cy,w,h) 转换为 (xmin, ymin, xmax, ymax)。
    """
    if isinstance(raw_preds, tuple):
        pred_tensor = raw_preds[0].cpu().squeeze(0)  # 原始 shape: [C, N]
    else:
        pred_tensor = raw_preds.cpu().squeeze(0)
    # print("[DEBUG] raw_preds[0] 原始 shape:", pred_tensor.shape)

    # 截取前 18 个通道：4用于边框，14用于类别logits
    pred_tensor = pred_tensor[:18, :]
    # print("[DEBUG] 截取后 pred_tensor shape:", pred_tensor.shape)

    # 提取边框数据 (cx,cy,w,h)
    boxes_data = pred_tensor[:4, :].permute(1, 0)  # shape: [N, 4]
    # 提取类别 logits 并应用 sigmoid 激活
    class_logits = pred_tensor[4:, :].permute(1, 0)  # shape: [N, 14]
    class_conf = class_logits.sigmoid()  # 将 logits 转换为概率

    # print("[DEBUG] boxes_data: min {:.4f}, max {:.4f}, mean {:.4f}".format(
    #     boxes_data.min().item(), boxes_data.max().item(), boxes_data.mean().item()))

    # 判断是否为归一化值（如果最大值<=1，则认为是归一化的）
    if boxes_data.max() <= 1:
        # print("[DEBUG] 检测框数据为归一化格式")
        cx = boxes_data[:, 0] * target_w
        cy = boxes_data[:, 1] * target_h
        w = boxes_data[:, 2] * target_w
        h = boxes_data[:, 3] * target_h
    else:
        # print("[DEBUG] 检测框数据为像素格式")
        cx = boxes_data[:, 0]
        cy = boxes_data[:, 1]
        w = boxes_data[:, 2]
        h = boxes_data[:, 3]

    # 将中心格式转换为角点格式
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    boxes = torch.stack([x1, y1, x2, y2], dim=1)
    # print("[DEBUG] boxes 转换后: min {:.2f}, max {:.2f}, mean {:.2f}".format(
    #     boxes.min().item(), boxes.max().item(), boxes.mean().item()))
    return boxes, class_conf


def main():
    model_path = r'D:\Programming\Project\github\KonColle\KC\Models\yolov8_KC_model.pt'
    checkpoint = torch.load(model_path, map_location='cpu')
    detection_class_names = checkpoint.get('detection_class_names', None)
    classification_class_id_to_name = checkpoint.get('class_id_to_name', None)
    model = checkpoint['model']
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 设置检测类别名称
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
        print("未找到检测类别名称。")
    if detection_class_names:
        print("检测模型可识别的类别:")
        if isinstance(detection_class_names, dict):
            for class_id, class_name in detection_class_names.items():
                print(f"检测类别 {class_id}: {class_name}")
        elif isinstance(detection_class_names, list):
            for idx, name in enumerate(detection_class_names):
                print(f"检测类别 {idx}: {name}")
    if classification_class_id_to_name:
        print("\n分类模型可识别的类别:")
        for class_id, class_name in classification_class_id_to_name.items():
            print(f"分类类别 {class_id}: {class_name}")

    with mss.mss() as sct:
        monitor = sct.monitors[1]  # 主屏幕
        conf_thresh = 0.50  # 阈值恢复到正常值
        iou_thresh = 0.45
        target_w, target_h = 1120, 672  # 模型输入尺寸
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        last_print_time = time.time()

        while True:
            sct_img = sct.grab(monitor)
            frame = np.array(sct_img)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            orig_h, orig_w = frame.shape[:2]
            frame_resized = cv2.resize(frame, (target_w, target_h))
            img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(device, dtype=torch.float32) / 255.0
            img_tensor = (img_tensor - mean) / std

            with torch.no_grad():
                class_logits, checkbox_logits, raw_preds = model(img_tensor)

            # 调试输出 raw_preds 信息
            pred_tensor = raw_preds[0].cpu().squeeze(0)
            print("[DEBUG] raw_preds[0] shape:", pred_tensor.shape)
            print("[DEBUG] raw_preds[0] 最大值：", pred_tensor.max().item(),
                  "最小值：", pred_tensor.min().item(),
                  "均值：", pred_tensor.mean().item())

            # 解析检测框和类别概率（直接使用类别概率作为置信度）
            boxes, class_conf = process_raw_preds(raw_preds, target_w, target_h)
            if boxes is None:
                print("检测输出格式异常")
                continue

            print("[DEBUG] 解析后 boxes shape:", boxes.shape)
            # 调试 class_conf 信息
            print("[DEBUG] class_conf: min {:.4f}, max {:.4f}, mean {:.4f}".format(
                class_conf.min().item(), class_conf.max().item(), class_conf.mean().item()))

            # 对每个预测，取最大类别概率及对应类别索引
            class_scores, class_ids = torch.max(class_conf, dim=1)
            scores = class_scores  # 直接使用最大类别概率作为得分
            print("[DEBUG] scores: min {:.4f}, max {:.4f}, mean {:.4f}".format(
                scores.min().item(), scores.max().item(), scores.mean().item()))

            # 过滤低于阈值的检测
            mask = scores >= conf_thresh
            boxes = boxes[mask]
            scores = scores[mask]
            class_ids = class_ids[mask]
            print("[DEBUG] 经过阈值过滤后，剩余检测数:", boxes.shape[0])

            # 非极大值抑制（NMS）
            final_indices = []
            if boxes.shape[0] > 0:
                for cls in class_ids.unique():
                    cls_mask = class_ids == cls
                    cls_boxes = boxes[cls_mask]
                    cls_scores = scores[cls_mask]
                    keep = nms(cls_boxes, cls_scores, iou_threshold=iou_thresh)
                    cls_indices = torch.nonzero(cls_mask, as_tuple=False).squeeze(1)
                    final_indices.append(cls_indices[keep])
                final_indices = torch.cat(final_indices).unique() if final_indices else torch.tensor([],
                                                                                                     dtype=torch.long)
            else:
                final_indices = torch.tensor([], dtype=torch.long)
            print("[DEBUG] NMS后剩余检测数:", len(final_indices))

            # 绘制检测框：将模型输入尺寸下的框映射回原始屏幕尺寸
            annotated_frame = frame.copy()
            for idx in final_indices.tolist():
                x1, y1, x2, y2 = boxes[idx].tolist()
                x1 = int(x1 * (orig_w / target_w))
                y1 = int(y1 * (orig_h / target_h))
                x2 = int(x2 * (orig_w / target_w))
                y2 = int(y2 * (orig_h / target_h))
                cls_id = int(class_ids[idx].item())
                score_val = float(scores[idx].item())
                if detection_class_names is not None:
                    if isinstance(detection_class_names, dict):
                        cls_name = detection_class_names.get(cls_id, str(cls_id))
                    else:
                        cls_name = detection_class_names[cls_id] if cls_id < len(detection_class_names) else str(cls_id)
                else:
                    cls_name = str(cls_id)
                color = tuple(int(c) for c in np.random.RandomState(cls_id).randint(0, 255, size=3))
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                label = f"{cls_name} {score_val:.2f}"
                (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                tx1, ty1 = x1, y1 - th - 3
                if ty1 < 0:
                    ty1 = y1
                cv2.rectangle(annotated_frame, (tx1, ty1), (tx1 + tw + 2, ty1 + th), color, -1)
                text_color = (255, 255, 255) if (color[0] * 0.299 + color[1] * 0.587 + color[2] * 0.114) < 186 else (
                0, 0, 0)
                cv2.putText(annotated_frame, label, (x1 + 1, ty1 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)

            cv2.imshow("Screen Detection", annotated_frame)
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
