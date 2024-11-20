# 导入必要的库
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50  # 修改为 ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import pandas as pd
import os
import math
# 导入回调函数
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
import datetime
import PIL
import scipy
# 导入混合精度策略
from tensorflow.keras import mixed_precision

# 设置混合精度策略
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# 定义数据集路径和对应的类别标签
dataset_paths = {
    'main_menu': r'D:\Programming\Project\github\KonColle\Datasets\images\main_menu',
    'supply': r'D:\Programming\Project\github\KonColle\Datasets\images\supply',
    'mission_select': r'D:\Programming\Project\github\KonColle\Datasets\images\mission_select',
    'map_1': r'D:\Programming\Project\github\KonColle\Datasets\images\waters\map_1',
    'map_5': r'D:\Programming\Project\github\KonColle\Datasets\images\waters\map_5',
    'map_5_info': r'D:\Programming\Project\github\KonColle\Datasets\images\waters\map_5\map_info',
    'map_5_fleet': r'D:\Programming\Project\github\KonColle\Datasets\images\waters\map_5\map_fleet',
    'advance_on': r'D:\Programming\Project\github\KonColle\Datasets\images\waters\in_map\advance_on',
    'combat_rating': r'D:\Programming\Project\github\KonColle\Datasets\images\waters\in_map\combat_rating',
    'combat_result': r'D:\Programming\Project\github\KonColle\Datasets\images\waters\in_map\combat_result',
    'navigation': r'D:\Programming\Project\github\KonColle\Datasets\images\waters\in_map\navigation',
    # 可以添加更多类别
}

# 创建列表来存储文件路径和标签
file_paths = []
labels = []

for label, directory in dataset_paths.items():
    # 遍历指定目录（不包括子目录）
    for filename in os.listdir(directory):  # 只读取当前目录
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # 筛选图片文件
            file_paths.append(os.path.join(directory, filename))
            labels.append(label)

# 创建 DataFrame
data = pd.DataFrame({'filename': file_paths, 'class': labels})

# 打印每个类别的样本数量
print("每个类别的样本数量：")
print(data['class'].value_counts())

# ==================== 参数配置 ====================

# 图像的宽度和高度（恢复为您的原始分辨率）
img_width, img_height = 1111, 667

# 训练的批次大小和训练轮数
batch_size = 8
epochs = 30

# 类别数量（自动获取类别数量）
num_classes = len(dataset_paths)

# ==================== 数据预处理和增强 ====================

# 创建用于训练数据的数据增强生成器
train_datagen = ImageDataGenerator(
    rescale=1./255,
    brightness_range=[0.9, 1.1],
    zoom_range=0.05,
    validation_split=0.2
)

# 创建用于验证数据的生成器（不进行数据增强）
validation_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# ==================== 数据生成器 ====================

# 创建训练数据生成器
train_generator = train_datagen.flow_from_dataframe(
    dataframe=data,
    x_col='filename',
    y_col='class',
    target_size=(img_height, img_width),  # 注意顺序为 (height, width)
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

# 创建验证数据生成器
validation_generator = validation_datagen.flow_from_dataframe(
    dataframe=data,
    x_col='filename',
    y_col='class',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# 检查类别映射
print("类别索引映射:", train_generator.class_indices)

# ==================== 构建模型 ====================

# 加载预训练的 ResNet50 模型，不包含顶层的全连接层
base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(img_height, img_width, 3)
)

# 冻结部分层，解冻后面的层进行微调
for layer in base_model.layers[:-10]:
    layer.trainable = False
for layer in base_model.layers[-10:]:
    layer.trainable = True

# 添加自定义的顶层网络
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax', dtype='float32')(x)

# 定义最终的模型
model = Model(inputs=base_model.input, outputs=predictions)

# ==================== 编译模型 ====================
# 使用混合精度的优化器
optimizer = mixed_precision.LossScaleOptimizer(tf.keras.optimizers.Adam(learning_rate=1e-5))

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ==================== 设置回调函数 ====================

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    filepath='best_model.keras',
    monitor='val_loss',
    save_best_only=True
)

# ==================== 计算步骤数 ====================

steps_per_epoch = math.ceil(train_generator.samples / batch_size)
validation_steps = math.ceil(validation_generator.samples / batch_size)

# ==================== 训练模型 ====================

model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    callbacks=[tensorboard_callback, early_stopping, checkpoint]
)

# ==================== 保存模型 ====================

# 指定模型保存的路径
model_save_path = r'D:\Programming\Project\github\KonColle\KC\Models\model.keras'
# 确保保存路径存在
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

# 保存模型到指定路径
model.save(model_save_path)

print("模型已成功保存到:", model_save_path)

model.summary()