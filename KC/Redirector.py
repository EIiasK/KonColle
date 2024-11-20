import os
import json

# 路径配置
image_root_dir = r"D:\Programming\Project\github\KonColle\Datasets\images"
annotation_file = r"D:\Programming\Project\github\KonColle\Datasets\annotations\instances_Train.json"

# 加载 COCO 标签
with open(annotation_file, 'r') as f:
    annotations = json.load(f)

# 遍历所有图片，修正路径
updated_images = []
missing_files = []

for img in annotations['images']:
    file_name = img['file_name']  # 当前的 file_name 字段
    found = False

    # 在根目录下递归查找图片
    for root, _, files in os.walk(image_root_dir):
        if file_name in files:
            # 找到文件，更新路径为相对路径
            relative_path = os.path.relpath(os.path.join(root, file_name), image_root_dir)
            img['file_name'] = relative_path
            updated_images.append(relative_path)
            found = True
            break

    if not found:
        # 未找到的图片记录到 missing_files
        missing_files.append(file_name)

# 保存修复后的标签文件
fixed_annotation_file = annotation_file.replace(".json", "_fixed.json")
with open(fixed_annotation_file, 'w') as f:
    json.dump(annotations, f)

# 输出结果
if missing_files:
    print("以下文件未找到：")
    for missing in missing_files:
        print(missing)
else:
    print("所有路径均已修复！")

print(f"修复后的标签文件已保存到: {fixed_annotation_file}")
