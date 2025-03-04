import torch

def inspect_model(model_path):
    try:
        # 加载保存的模型字典
        model_info = torch.load(model_path, map_location='cpu')
    except Exception as e:
        print(f"加载模型时出错: {e}")
        return

    # 提取检测类别名称
    detection_class_names = model_info.get('detection_class_names', None)
    if detection_class_names:
        print("检测模型可以识别的所有类别及其标签：")
        if isinstance(detection_class_names, dict):
            # 如果检测类别名称是字典，按键值对遍历
            for class_id, class_name in detection_class_names.items():
                print(f"检测类别 {class_id}: {class_name}")
        elif isinstance(detection_class_names, list):
            # 如果检测类别名称是列表，使用枚举遍历
            for class_id, class_name in enumerate(detection_class_names):
                print(f"检测类别 {class_id}: {class_name}")
        else:
            print("检测类别名称的格式不正确。")
    else:
        print("未找到检测类别名称。")

    # 提取分类类别名称
    classification_class_id_to_name = model_info.get('class_id_to_name', None)
    if classification_class_id_to_name:
        print("\n分类模型可以识别的所有类别及其标签：")
        for class_id, class_name in classification_class_id_to_name.items():
            print(f"分类类别 {class_id}: {class_name}")
    else:
        print("未找到分类类别名称。")

    # 提取 `checkbox_head` 的权重并分析是否存在 `checkbox` 属性
    checkbox_head_state = model_info.get('checkbox_head', None)
    if checkbox_head_state:
        print("\n模型中的 `checkbox_head` 权重信息：")
        for key, value in checkbox_head_state.items():
            print(f"{key}: {value.shape}")

        # 判断哪些类别拥有 `checkbox` 属性
        print("\n检测每个类别的 `checkbox` 属性的权重值：")
        checkbox_weights = checkbox_head_state.get('weight', None)  # 获取权重

        if checkbox_weights is not None:
            # 查看每个类别的权重值
            for class_id in range(len(detection_class_names)):
                weight_value = checkbox_weights[0, class_id].item()  # 获取权重值

                # 输出每个类别的权重值
                print(f"类别 {class_id} ({detection_class_names[class_id]}): 权重值 = {weight_value}")

                # 通过权重值来判断是否有 checkbox 属性
                if abs(weight_value) > 0.01:  # 如果权重绝对值大于0.01，认为该类别有 checkbox 属性
                    print(f"检测类别 {class_id} ({detection_class_names[class_id]}) 具有 `checkbox` 属性。")
                else:
                    print(f"检测类别 {class_id} ({detection_class_names[class_id]}) 没有 `checkbox` 属性。")
        else:
            print("未能提取 `checkbox_head` 的权重。")
    else:
        print("未找到 `checkbox_head` 信息，无法显示 `checkbox` 属性。")

def main():
    # 请确保路径正确，并且模型文件存在
    model_path = r'D:\Programming\Project\github\KonColle\KC\Models\yolov8_KC_model.pt'
    inspect_model(model_path)

if __name__ == "__main__":
    main()



# r'D:\Programming\Project\github\KonColle\KC\Models\yolov8_KC_model.pt'