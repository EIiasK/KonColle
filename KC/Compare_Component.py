import json
import os

# 路径配置
image_dir = r"D:\Programming\Project\github\KonColle\Datasets\images"
annotation_file = r"D:\Programming\Project\github\KonColle\Datasets\annotations\instances_Train_fixed.json"

# 加载 COCO 标签
with open(annotation_file, 'r') as f:
    annotations = json.load(f)

missing_files = []

# 检查文件是否存在
for img in annotations['images']:
    file_path = os.path.join(image_dir, img['file_name'])
    if not os.path.exists(file_path):
        missing_files.append(file_path)

if missing_files:
    print("以下文件缺失：")
    for file in missing_files:
        print(file)
else:
    print("所有文件都存在且路径有效。")
