import os
import torch
from torch.utils.tensorboard import SummaryWriter
from ultralytics import YOLO
import yaml

# 加载类别名称列表，每行一个类别
def load_names(names_path):
    with open(names_path, 'r', encoding='utf-8') as f:
        names = [line.strip() for line in f if line.strip()]
    return names

def main():
    # ---------------- 数据集路径配置 ----------------
    dataset_root   = r"D:\Programming\Project\github\KonColle\Detection_Data"
    # 下面的两行用于“拆分脚本”后，确定 train/val 子目录存在
    images_train_dir = os.path.join(dataset_root, "images", "train")
    images_val_dir   = os.path.join(dataset_root, "images", "val")
    labels_train_dir = os.path.join(dataset_root, "labels", "train")
    labels_val_dir   = os.path.join(dataset_root, "labels", "val")
    names_path       = os.path.join(dataset_root, "names.txt")

    # ---------------- 模型与日志路径 ----------------
    model_save_path = r"D:\Programming\Project\github\KonColle\KC\Models\KC_Detection_model.pt"
    log_dir         = r"D:\Programming\Project\github\KonColle\KC\Logs"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    # 加载类别
    det_names = load_names(names_path)
    num_det_classes = len(det_names)
    print(f"检测类别数: {num_det_classes}, 类别列表: {det_names}")

    # ---------------- 生成 data.yaml（已修改为 train/val 子目录） ----------------
    data_cfg = {
        'path': dataset_root,
        'train': 'images/train',   # 修改为 images/train
        'val':   'images/val',     # 修改为 images/val
        'nc':    num_det_classes,
        'names': det_names
    }
    data_yaml_path = os.path.join(log_dir, 'data.yaml')
    with open(data_yaml_path, 'w') as f:
        yaml.dump(data_cfg, f)
    print(f"已生成 data.yaml：{data_yaml_path}")

    # ---------------- 初始化 YOLOv8 模型 ----------------
    yolo_model_path = r"D:\Programming\Project\github\KonColle\KC\Models\YOLOv8\yolov8n.pt"
    model = YOLO(yolo_model_path)

    # ---------------- 启动训练 ----------------
    model.train(
        data=data_yaml_path,
        epochs=25,
        imgsz=960,
        batch=16,
        workers=0,
        cache=True,
        optimizer='SGD',
        lr0=1e-2,
        momentum=0.937,
        weight_decay=5e-4,
        verbose=True,
        degrees=0.0,
        translate=0.1,
        scale=0.7,
        shear=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        box=7.5,
        cls=1.0,
        dfl=1.5
    )

    # ---------------- 评估并输出指标 ----------------
    results = model.val()
    m = results.box

    mean_p   = m.mp     # numpy.float64 类型
    mean_r   = m.mr     # numpy.float64 类型
    map50    = m.map50  # numpy.float64 类型
    map5095  = m.map    # numpy.float64 类型

    print(f"Precision (mean): {mean_p:.4f}, Recall (mean): {mean_r:.4f}, "
          f"mAP@50: {map50:.4f}, mAP@50-95: {map5095:.4f}")
    # ---------------- 保存最终模型 ----------------
    model.save(model_save_path)
    print(f"训练完毕，模型已保存至: {model_save_path}")

    writer.close()

if __name__ == '__main__':
    main()
