import os
import json
import copy
from pathlib import Path

def generate_coco_labels(image_dir, base_coco_file, output_coco_file):
    """
    根据基准 COCO 标签文件为文件夹内的所有图片生成新的 COCO 1.0 标签文件。

    :param image_dir: 图片文件夹路径。
    :param base_coco_file: 基准 COCO 标签文件路径，仅包含一个图片的标签信息。
    :param output_coco_file: 输出的 COCO 标签文件路径。
    """
    # 读取基准 COCO 文件
    with open(base_coco_file, 'r') as f:
        base_coco_data = json.load(f)

    # 验证基准文件是否包含一个图片和相关标签
    if len(base_coco_data['images']) != 1:
        raise ValueError("基准 COCO 文件必须仅包含一个图片的标签信息。")

    base_image_info = base_coco_data['images'][0]
    base_annotations = base_coco_data['annotations']

    # 获取文件夹内所有图片文件名
    image_files = [
        file for file in os.listdir(image_dir)
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
    ]

    if not image_files:
        raise ValueError(f"文件夹 {image_dir} 中未找到任何图片文件。")

    # 初始化新的 COCO 数据结构
    new_coco_data = {
        "info": base_coco_data.get("info", {}),
        "licenses": base_coco_data.get("licenses", []),
        "images": [],
        "annotations": [],
        "categories": base_coco_data.get("categories", [])
    }

    # 遍历文件夹内的图片，生成 COCO 标签
    annotation_id = 1
    for image_id, image_file in enumerate(image_files, start=1):
        # 创建新的图片信息
        image_path = Path(image_dir) / image_file
        new_image_info = copy.deepcopy(base_image_info)
        new_image_info['id'] = image_id
        new_image_info['file_name'] = image_file
        new_image_info['width'] = base_image_info['width']  # 可根据实际需求修改
        new_image_info['height'] = base_image_info['height']  # 可根据实际需求修改
        new_coco_data['images'].append(new_image_info)

        # 创建新的注释信息
        for base_annotation in base_annotations:
            new_annotation = copy.deepcopy(base_annotation)
            new_annotation['id'] = annotation_id
            new_annotation['image_id'] = image_id
            new_coco_data['annotations'].append(new_annotation)
            annotation_id += 1

    # 保存新的 COCO 文件
    with open(output_coco_file, 'w') as f:
        json.dump(new_coco_data, f, indent=4)
    print(f"新的 COCO 标签文件已保存至: {output_coco_file}")


# 示例用法
if __name__ == "__main__":
    # 输入图片文件夹路径
    image_directory = r"D:\Programming\Project\github\KonColle\Datasets\Applier\example\waters\in_map\advance_on"
    # 基准 COCO 标签文件路径
    base_coco_label_file = r"D:\Programming\Project\github\KonColle\Datasets\Applier\example\waters\in_map\advance_on\advance_on_standard.json"
    # 输出新的 COCO 标签文件路径
    output_coco_label_file = r"D:\Programming\Project\github\KonColle\Datasets\Applier\output\waters\in_map\advance_on\advance_on.json"

    generate_coco_labels(image_directory, base_coco_label_file, output_coco_label_file)
