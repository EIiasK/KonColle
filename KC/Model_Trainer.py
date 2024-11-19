# 导入必要的库
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import pandas as pd
import os
#导入 TensorBoard 回调函数
from tensorflow.keras.callbacks import TensorBoard
import datetime

# 定义数据集路径和对应的类别标签
dataset_paths = {
    'main_menu': 'D:\Programming\Project\github\KonColle\Datasets\images\main_menu',  # 替换为实际路径
    'supply': 'D:\Programming\Project\github\KonColle\Datasets\images\supply',
    'mission_select': 'D:\Programming\Project\github\KonColle\Datasets\images\mission_select',
    'map_1': 'D:\Programming\Project\github\KonColle\Datasets\images\waters\map_1',
    'map_5': 'D:\Programming\Project\github\KonColle\Datasets\images\waters\map_5',
    'map_5_info': 'D:\Programming\Project\github\KonColle\Datasets\images\waters\map_5\map_info',
    'map_5_fleet': 'D:\Programming\Project\github\KonColle\Datasets\images\waters\map_5\map_fleet',
    'advance_on': 'D:\Programming\Project\github\KonColle\Datasets\images\waters\in_map\advance_on',
    'combat_rating': 'D:\Programming\Project\github\KonColle\Datasets\images\waters\in_map\combat_rating',
    'combat_result': 'D:\Programming\Project\github\KonColle\Datasets\images\waters\in_map\combat_result',
    'navigation': 'D:\Programming\Project\github\KonColle\Datasets\images\waters\in_map\navigation',
    #'': '',
    # 可以添加更多类别
}

# 创建列表来存储文件路径和标签
file_paths = []
labels = []

for label, directory in dataset_paths.items():
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_paths.append(os.path.join(root, filename))
                labels.append(label)

# 创建 DataFrame
data = pd.DataFrame({'filename': file_paths, 'class': labels})

# ==================== 参数配置 ====================

# 图像的宽度和高度
img_width, img_height = 1111, 667

# 训练的批次大小和训练轮数
batch_size = 32
epochs = 10  # 您可以根据需要调整训练轮数

# 类别数量（请根据您的实际类别数进行修改）
num_classes = 5  # 例如，如果有5个不同的界面或按钮

# ==================== 数据预处理和增强 ====================

# 创建用于训练数据的数据增强生成器
train_datagen = ImageDataGenerator(
    rescale=1./255,  # 将像素值缩放到0-1之间
    brightness_range=[0.8, 1.2],  # 随机调整亮度
    zoom_range=0.1,  # 随机缩放图像
    validation_split=0.2  # 预留20%数据用于验证
    # 由于图像方向固定，通常不进行翻转
    # horizontal_flip=False,
    # vertical_flip=False,
    # 其他可能的增强方式（根据需要启用）
    # width_shift_range=0.05,
    # height_shift_range=0.05,
)

# 创建用于验证数据的生成器（不进行数据增强）
validation_datagen = ImageDataGenerator(
    rescale=1./255,            # 只进行归一化
    validation_split=0.2       # 与训练数据相同的验证集划分
)

# ==================== 数据生成器 ====================

# 创建训练数据生成器
train_generator = train_datagen.flow_from_dataframe(
    dataframe=data,
    x_col='filename',
    y_col='class',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',  # 使用训练集
    shuffle=True
)

# 创建验证数据生成器
validation_generator = validation_datagen.flow_from_dataframe(
    dataframe=data,
    x_col='filename',
    y_col='class',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',  # 使用验证集
    shuffle=False
)
# ==================== 构建模型 ====================

# 加载预训练的MobileNet模型，不包含顶层的全连接层
base_model = MobileNet(
    weights='imagenet',  # 使用在ImageNet上预训练的权重
    include_top=False,   # 不包含顶层
    input_shape=(img_width, img_height, 3)  # 输入图像的形状
)

# 冻结预训练模型的所有层，以防止它们在训练过程中更新
for layer in base_model.layers:
    layer.trainable = False

# 添加自定义的顶层网络
x = base_model.output
x = GlobalAveragePooling2D()(x)  # 添加全局平均池化层
x = Dense(1024, activation='relu')(x)  # 添加全连接层
predictions = Dense(num_classes, activation='softmax')(x)  # 输出层，使用softmax激活函数进行多分类

# 定义最终的模型
model = Model(inputs=base_model.input, outputs=predictions)

# ==================== 编译模型 ====================

# 编译模型，指定损失函数、优化器和评价指标
model.compile(
    optimizer=tf.keras.optimizers.Adam(),  # 使用Adam优化器
    loss='categorical_crossentropy',       # 多分类的对数损失函数
    metrics=['accuracy']                   # 评价指标为准确率
)

# ==================== 训练模型 ====================

# 开始训练模型
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,  # 每轮的批次数量
    epochs=epochs,  # 训练的轮数
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size  # 验证的批次数量
)

# ==================== 保存模型 ====================

# 指定模型保存的路径（请填写您要保存模型的路径）
model_save_path = 'path_to_save_model/model.h5'  # 示例：'saved_models/model.h5'

# 保存模型到指定路径
model.save(model_save_path)

print("模型已成功保存到:", model_save_path)
