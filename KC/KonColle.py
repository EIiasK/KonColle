import cv2
import numpy as np
import time
from PIL import ImageGrab
import torch
from ultralytics import YOLO

def main():
    # 1. 配置设备与模型路径
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = r"D:\Programming\Project\github\KonColle\KC\Models\KC_Detection_model.pt"
    model = YOLO(ckpt_path)
    model.to(device)
    model.model.eval()

    det_names = [model.names[i] for i in sorted(model.names.keys())]
    print("检测类别：", det_names)

    # 4. 定义要截取屏幕的 ROI 坐标
    X, Y, W, H = 0, 312, 1512, 952

    # 5. 先创建一个可调节大小的窗口
    window_name = "实时目标检测"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # 6. 希望显示成原来的一半大小（你可以改为 0.5、0.6、0.7 之类任意比例）
    scale = 0.5
    display_W = int(W * scale)
    display_H = int(H * scale)
    cv2.resizeWindow(window_name, display_W, display_H)

    # 7. 循环抓屏并推理
    target_fps = 5
    frame_interval = 1.0 / target_fps

    while True:
        t0 = time.time()

        screenshot = ImageGrab.grab()
        full_img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        crop = full_img[Y:Y+H, X:X+W]

        results = model(crop, conf=0.5, imgsz=(1120, 672), device=device)[0]
        boxes   = results.boxes.xyxy.cpu().numpy()
        confs   = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)

        annotated = crop.copy()
        for (x1, y1, x2, y2), conf, cls_id in zip(boxes, confs, classes):
            name  = det_names[cls_id] if 0 <= cls_id < len(det_names) else str(cls_id)
            label = f"{name} {conf:.2f}"
            # 框用黑色
            cv2.rectangle(annotated,
                          (int(x1), int(y1)),
                          (int(x2), int(y2)),
                          (0, 255, 0), 2)  # 黑色 (0,0,0)

            # 背景用黑色，文字用粉色 (B=255, G=0, R=255)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(annotated,
                          (int(x1), int(y1) - th - 4),
                          (int(x1) + tw, int(y1)),
                          (0, 255, 0), -1)
            cv2.putText(annotated, label, (int(x1), int(y1) - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        # 8. 对 annotated 做缩放，再显示
        small = cv2.resize(annotated, (display_W, display_H))
        cv2.imshow(window_name, small)

        if cv2.waitKey(1) == 27:
            break
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

        elapsed = time.time() - t0
        delay = frame_interval - elapsed
        if delay > 0:
            end_t = time.time() + delay
            while time.time() < end_t:
                cv2.waitKey(1)
                time.sleep(0.005)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
