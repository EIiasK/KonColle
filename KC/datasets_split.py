import os
import shutil
import random

# 请根据实际路径修改
dataset_root = r"D:\Programming\Project\github\KonColle\Detection_Data"
images_dir   = os.path.join(dataset_root, "images")
labels_dir   = os.path.join(dataset_root, "labels")

# 新建子文件夹 images/train, images/val, labels/train, labels/val
os.makedirs(os.path.join(images_dir, "train"), exist_ok=True)
os.makedirs(os.path.join(images_dir, "val"), exist_ok=True)
os.makedirs(os.path.join(labels_dir, "train"), exist_ok=True)
os.makedirs(os.path.join(labels_dir, "val"), exist_ok=True)

# 假设所有图片后缀是 .jpg（如果有 .png 一并处理）
all_images = [f for f in os.listdir(images_dir) if f.lower().endswith(".jpg") or f.lower().endswith(".png")]

# 打乱顺序
random.seed(42)
random.shuffle(all_images)

# 按 80%:20% 分
split_idx = int(len(all_images) * 0.8)
train_imgs = all_images[:split_idx]
val_imgs   = all_images[split_idx:]

# 将文件从 images/ 移动到 images/train 或 images/val，同时对应地移动 labels/*.txt
for img_name in train_imgs:
    base_name = os.path.splitext(img_name)[0]
    # 移动图片
    shutil.move(os.path.join(images_dir, img_name),
                os.path.join(images_dir, "train", img_name))
    # 移动该图片对应的标签
    lbl_name = base_name + ".txt"
    src_lbl = os.path.join(labels_dir, lbl_name)
    if os.path.exists(src_lbl):
        shutil.move(src_lbl, os.path.join(labels_dir, "train", lbl_name))

for img_name in val_imgs:
    base_name = os.path.splitext(img_name)[0]
    shutil.move(os.path.join(images_dir, img_name),
                os.path.join(images_dir, "val", img_name))
    lbl_name = base_name + ".txt"
    src_lbl = os.path.join(labels_dir, lbl_name)
    if os.path.exists(src_lbl):
        shutil.move(src_lbl, os.path.join(labels_dir, "val", lbl_name))

print(f"训练集: {len(train_imgs)} 张图片；验证集: {len(val_imgs)} 张图片。")
