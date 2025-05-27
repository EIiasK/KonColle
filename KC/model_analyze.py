import torch
from Model_Trainer_YOLOv8 import AttrDict, YOLOv8WithClassification
from ultralytics import YOLO
import os


def inspect_model(model_path):
    try:
        model_info = torch.load(model_path, map_location='cpu')
    except Exception as e:
        print(f"加载模型时出错: {e}")
        return
    if isinstance(model_info, dict):
        detection_class_names = model_info.get('detection_class_names', None)
        classification_class_id_to_name = model_info.get('class_id_to_name', None)
        checkbox_head_state = model_info.get('checkbox_head', None)
    else:
        detection_class_names = getattr(model_info, 'detection_class_names', None)
        classification_class_id_to_name = getattr(model_info, 'class_id_to_name', None)
        checkbox_head_state = getattr(model_info, 'checkbox_head', None)

    # 展示检测类别及其标签
    if detection_class_names:
        print("检测模型可以识别的所有类别及其标签：")
        if isinstance(detection_class_names, dict):
            for class_id, class_name in detection_class_names.items():
                print(f"检测类别 {class_id}: {class_name}")
        elif isinstance(detection_class_names, list):
            for class_id, class_name in enumerate(detection_class_names):
                print(f"检测类别 {class_id}: {class_name}")
        else:
            print("检测类别名称的格式不正确。")
    else:
        print("未找到检测类别名称。")

    # 展示分类类别及其标签
    if classification_class_id_to_name:
        print("\n分类模型可以识别的所有类别及其标签：")
        for class_id, class_name in classification_class_id_to_name.items():
            print(f"分类类别 {class_id}: {class_name}")
    else:
        print("未找到分类类别名称。")

    # 打印 checkbox head 的权重信息（供参考）
    if checkbox_head_state:
        try:
            state_dict = checkbox_head_state.state_dict()
            print("\n模型中的 `checkbox_head` 权重信息：")
            for key, value in state_dict.items():
                print(f"{key}: {value.shape}")
        except Exception as e:
            print(f"处理 `checkbox_head` 时出错: {e}")
    else:
        print("未找到 `checkbox_head` 信息。")


def main():
    # 请将以下路径修改为你实际的模型保存路径
    model_path = r'D:\Programming\Project\github\KonColle\KC\Models\yolov8_KC_model.pt'
    inspect_model(model_path)


if __name__ == "__main__":
    main()
