# 导入必要的库
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import os

# ==================== 参数配置 ====================

# 图像的宽度和高度
img_width, img_height = 224, 224

# 训练的批次大小和训练轮数
batch_size = 32
epochs = 10  # 您可以根据需要调整训练轮数

# 类别数量（请根据您的实际类别数进行修改）
num_classes = 5  # 例如，如果有5个不同的界面或按钮

# ==================== 数据路径 ====================

# 训练数据集的目录路径（请填写您的训练数据集路径）
train_data_dir = 'path_to_train_dataset'  # 示例：'data/train'

# 验证数据集的目录路径（请填写您的验证数据集路径）
validation_data_dir = 'path_to_validation_dataset'  # 示例：'data/validation'

# ==================== 数据预处理和增强 ====================

# 创建用于训练数据的数据增强生成器
train_datagen = ImageDataGenerator(
    rescale=1./255,  # 将像素值缩放到0-1之间
    brightness_range=[0.8, 1.2],  # 随机调整亮度
    zoom_range=0.1,  # 随机缩放图像
    # 由于图像方向固定，通常不进行翻转
    # horizontal_flip=False,
    # vertical_flip=False,
    # 其他可能的增强方式（根据需要启用）
    # width_shift_range=0.05,
    # height_shift_range=0.05,
)

# 创建用于验证数据的生成器（不进行数据增强）
validation_datagen = ImageDataGenerator(rescale=1./255)

# ==================== 数据生成器 ====================

# 生成训练数据
train_generator = train_datagen.flow_from_directory(
    train_data_dir,  # 训练数据目录
    target_size=(img_width, img_height),  # 调整图像大小
    batch_size=batch_size,
    class_mode='categorical'  # 如果是多分类问题
)

# 生成验证数据
validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,  # 验证数据目录
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
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
