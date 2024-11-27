import json


def extract_single_image_coco(input_coco_file, output_coco_file):
    """
    从COCO文件中提取第一个图片及其标签信息，并保存为新的COCO文件。

    :param input_coco_file: 原始COCO文件路径
    :param output_coco_file: 提取后的COCO文件保存路径
    """
    with open(input_coco_file, 'r') as f:
        coco_data = json.load(f)

    # 提取第一个图片
    if not coco_data["images"]:
        raise ValueError("COCO文件中没有图片信息。")

    first_image = coco_data["images"][0]
    first_image_id = first_image["id"]

    # 提取与该图片相关的标签
    related_annotations = [
        ann for ann in coco_data["annotations"] if ann["image_id"] == first_image_id
    ]

    # 构建新的COCO数据结构
    new_coco_data = {
        "info": coco_data.get("info", {}),
        "licenses": coco_data.get("licenses", []),
        "categories": coco_data.get("categories", []),
        "images": [first_image],  # 仅包含第一个图片
        "annotations": related_annotations  # 关联的标签
    }

    # 保存到新的文件
    with open(output_coco_file, 'w') as f:
        json.dump(new_coco_data, f, indent=4)

    print(f"提取的COCO标签已保存到: {output_coco_file}")


# 示例用法
input_coco_path = r"D:\Programming\Project\github\KonColle\Datasets\Applier\example\main_menu\main_menu.json"  # 替换为你的COCO文件路径
output_coco_path = r"D:\Programming\Project\github\KonColle\Datasets\Applier\example\main_menu\main_menu_standard.json"  # 输出路径
extract_single_image_coco(input_coco_path, output_coco_path)
