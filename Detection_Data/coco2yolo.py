#!/usr/bin/env python3
import os, json

# ====== 配置区 ======
# COCO 格式的标注文件路径
json_path   = r"D:\Programming\Project\github\KonColle\Pics&annotations\annotations\KonColle_coco.json"
# 对应的图片文件夹根目录
images_dir  = r"D:\Programming\Project\github\KonColle\Detection_Data\images"
# 输出的 labels 文件夹
labels_dir  = r"D:\Programming\Project\github\KonColle\Detection_Data\labels"

# ====== 读取 JSON ======
with open(json_path, 'r', encoding='utf-8') as f:
    coco = json.load(f)

# 构建 image_id -> (file_name, width, height)
images_info = { img['id']: (img['file_name'], img['width'], img['height']) 
                for img in coco['images'] }

# 构建 image_id -> list of annotations
ann_dict = {}
for ann in coco['annotations']:
    img_id = ann['image_id']
    ann_dict.setdefault(img_id, []).append(ann)

# COCO JSON 中 categories 已经按 id 顺序给出了 name
# 这里不需要 names.txt，训练时 data.yaml 里直接用 names 列表即可

# ====== 转换并写入 YOLO TXT ======
for img_id, (fname, w, h) in images_info.items():
    anns = ann_dict.get(img_id, [])
    # 如果某些图片没有任何目标，也生成一个空文件，保证 Image<->Label 一一对应
    label_path = os.path.join(labels_dir, os.path.splitext(fname)[0] + ".txt")
    with open(label_path, 'w', encoding='utf-8') as out:
        for ann in anns:
            # COCO bbox: [xmin, ymin, width, height]
            xmin, ymin, bw, bh = ann['bbox']
            x_c = xmin + bw/2
            y_c = ymin + bh/2
            # 归一化
            x_c /= w
            y_c /= h
            bw  /= w
            bh  /= h
            # COCO 的 category_id 从 1 开始，YOLO TXT 要从 0 开始
            class_id = ann['category_id'] - 1
            out.write(f"{class_id} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}\n")

print(f"Converted {len(images_info)} images. Labels saved to: {labels_dir}")
