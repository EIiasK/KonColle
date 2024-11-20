import os
import json

# 设置路径
annotation_file = r"D:\Programming\Project\github\KonColle\Datasets\annotations\instances_Train.json"
images_root = r"D:\Programming\Project\github\KonColle\Datasets\images"

# 加载 COCO 格式标签文件
with open(annotation_file, 'r') as f:
    annotations = json.load(f)

# 更新 file_name 字段
for image in annotations['images']:
    file_name = image['file_name']  # 当前 file_name，可能只是文件名
    # 查找文件在 images_root 目录下的相对路径
    for root, _, files in os.walk(images_root):
        if file_name in files:
            # 更新 file_name 为相对于 images_root 的相对路径
            relative_path = os.path.relpath(os.path.join(root, file_name), images_root)
            image['file_name'] = relative_path
            break

# 保存更新后的标签文件
fixed_annotation_file = annotation_file.replace('.json', '_fixed.json')
with open(fixed_annotation_file, 'w') as f:
    json.dump(annotations, f)
print(f"更新后的标签文件已保存到: {fixed_annotation_file}")
