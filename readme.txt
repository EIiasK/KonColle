Datasets文件夹
	---annotations文件夹   含有标签文件COCO1.0格式
	---images文件夹          为图片数据集，以文件夹名称作为分类类别基准划为多个文件夹
	---Applier文件夹          是快速依据模板打标签的暂存文件夹
KC文件夹
	---Game-interact文件夹      预置的屏幕捕获程序，暂未用到
	---Log文件夹                      保存模型训练时Tensorboard的记录日志
	---Models文件夹                含训练完的模型，以及放置在文件夹中的YOLOv8n预训练权重的模型
	---Coco_Copier.py    		根据基准 COCO 标签文件为文件夹内的所有图片生成新的 COCO 1.0 标签文件。（批量打标签）
	---Coco_Example_get.py    从COCO文件中提取第一个图片的标签信息，作为基准标签文件
	---Compare_Component.py     检查COCO标签文件中的图片file name是否跟其实际相对路径一致，在以文件为分类名的构建中很关键
	---KonColle.py			游戏脚本的主程序，未完成
	---Model_Trainer_Detr.py  早期利用Detr模型训练的程序，由于数据量需求过大弃用
	---Model_Trainer_YOLOv8.py    正在使用的模型训练器
	---Redirector.py                将COCO文件的file name字段更改为图片的相对路径
	---test.py                          测试模型支持的分类类别与检测标签
	---Course_design.py         通过指定数据集中的图片并调用模型来识别验证模型的准确性

运行环境：
python 3.12

库需求：
torch
torchvision
tqdm
Pillow
pycocotools
scikit-learn
ultralytics
asyncio
websockets