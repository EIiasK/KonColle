import cv2
import numpy as np
from PIL import ImageGrab

# 1. 抓取全屏
screenshot = ImageGrab.grab()
img_full = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
h_full, w_full = img_full.shape[:2]

# 2. 计算预览缩放比例（不放大，只做必要的缩小）
max_preview_w, max_preview_h = 1000, 700
scale = min(max_preview_w / w_full, max_preview_h / h_full, 1.0)
preview_w, preview_h = int(w_full * scale), int(h_full * scale)

# 3. 生成并显示可调整大小的预览窗口
img_preview = cv2.resize(img_full, (preview_w, preview_h), interpolation=cv2.INTER_AREA)
cv2.namedWindow("请框选区域", cv2.WINDOW_NORMAL)
cv2.resizeWindow("请框选区域", preview_w, preview_h)

# 4. 用鼠标在预览图上选 ROI，返回的是预览图坐标
x_p, y_p, w_p, h_p = cv2.selectROI("请框选区域", img_preview, showCrosshair=True, fromCenter=False)
cv2.destroyWindow("请框选区域")

# 5. 将预览图坐标映射回原始全屏坐标
x = int(x_p / scale)
y = int(y_p / scale)
w = int(w_p / scale)
h = int(h_p / scale)
print(f"全屏坐标： x={x}, y={y}, 宽={w}, 高={h}")

# 6. 按这个坐标裁剪全屏图
roi = img_full[y:y + h, x:x + w].copy()

# 7. 显示裁剪结果验证
cv2.imshow("ROI", roi)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 8. 后续你就可以用 roi 做模型推理了：
#    results = model(roi, conf=…, imgsz=(w, h), …)
