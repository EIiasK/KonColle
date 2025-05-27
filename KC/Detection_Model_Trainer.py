import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms
from torchvision.datasets import CocoDetection
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import copy
from collections import Counter
from PIL import Image
from torchvision.ops import box_iou
from pycocotools.coco import COCO
from sklearn.model_selection import train_test_split
import json
from collections import OrderedDict
from ultralytics import YOLO
from ultralytics.utils.loss import v8DetectionLoss

# 抑制冗长日志输出
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# 支持属性访问的字典类
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

# ==================== 自定义函数 ====================

def custom_collate_fn(batch):
    images = [item[0] for item in batch]
    detection_targets = [item[1] for item in batch]
    return images, detection_targets

# 提取COCO标注文件中的检测类别名称，返回类别ID到名称的映射
def extract_detection_classes(annotation_file):
    with open(annotation_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    categories = sorted(data.get('categories', []), key=lambda x: x['id'])
    mapping = OrderedDict()
    for idx, cat in enumerate(categories):
        mapping[idx] = cat['name']
    return mapping

# 获取各类别下的图片路径，用于划分数据集
def get_category_image_paths(base_dir):
    paths = {}
    for root, dirs, files in os.walk(base_dir):
        imgs = [f for f in files if f.lower().endswith(('.png','.jpg','.jpeg','.bmp','.gif'))]
        rel = os.path.relpath(root, base_dir)
        cls = os.path.basename(rel)
        if imgs:
            paths[cls] = [os.path.join(rel, f) for f in imgs]
    return paths

# 分割训练/验证集
def split_dataset(category_image_paths, test_size=0.2, random_state=42):
    train_imgs, val_imgs = [], []
    for cls, imgs in category_image_paths.items():
        if not imgs: continue
        t, v = train_test_split(imgs, test_size=test_size, random_state=random_state)
        train_imgs.extend(t)
        val_imgs.extend(v)
    return train_imgs, val_imgs

# 根据图片列表过滤原COCO注释，生成新的训练/验证注释文件
def filter_coco_annotations(annotation_file, image_files, output_file):
    with open(annotation_file, 'r') as f:
        coco = json.load(f)
    fname_to_id = {img['file_name']: img['id'] for img in coco['images']}
    sel_ids = set()
    sel_imgs = []
    for img in coco['images']:
        if img['file_name'] in image_files:
            sel_imgs.append(img)
            sel_ids.add(img['id'])
    sel_anns = [ann for ann in coco['annotations'] if ann['image_id'] in sel_ids]
    newcoco = {
        'info': coco.get('info', {}),
        'licenses': coco.get('licenses', []),
        'images': sel_imgs,
        'annotations': sel_anns,
        'categories': coco.get('categories', [])
    }
    with open(output_file, 'w') as f:
        json.dump(newcoco, f)

# ==================== 自定义数据集类 ====================
class CustomCocoDataset(CocoDetection):
    def __init__(self, root, annotation_file, transform=None, image_files=None):
        super().__init__(root=root, annFile=annotation_file)
        self.transform = transform
        if image_files is not None:
            fname_to_id = {info['file_name']: id_ for id_, info in self.coco.imgs.items()}
            self.ids = [fname_to_id[f] for f in image_files if f in fname_to_id]
        else:
            self.ids = list(self.coco.imgs.keys())

    def __getitem__(self, idx):
        img, ann = super().__getitem__(idx)
        if self.transform:
            img = self.transform(img)
        info = self.coco.imgs[self.ids[idx]]
        w, h = info['width'], info['height']
        boxes, labels, area, iscrowd = [], [], [], []
        for obj in ann:
            xmin, ymin, bw, bh = obj['bbox']
            x_c = (xmin + bw/2) / w
            y_c = (ymin + bh/2) / h
            boxes.append([x_c, y_c, bw/w, bh/h])
            labels.append(obj['category_id'] - 1)
            area.append(obj['area'])
            iscrowd.append(obj.get('iscrowd', 0))
        target = {
            'bboxes': torch.as_tensor(boxes, dtype=torch.float32),
            'cls': torch.as_tensor(labels, dtype=torch.int64),
            'image_id': torch.tensor([self.ids[idx]]),
            'area': torch.as_tensor(area, dtype=torch.float32),
            'iscrowd': torch.as_tensor(iscrowd, dtype=torch.int64)
        }
        return img, target

# ==================== 训练主流程 ====================
def main():
    # 参数配置
    base_dir = r"D:\Programming\Project\github\KonColle\Datasets\images"
    annotation_file = r"D:\Programming\Project\github\KonColle\Datasets\annotations\KonColle_coco_fixed.json"
    img_width, img_height = 1120, 672
    batch_size = 8
    epochs = 30
    learning_rate = 1e-4
    model_save_path = r"D:\Programming\Project\github\KonColle\KC\Models\KC_Detection_model.pt"
    log_dir = r"D:\Programming\Project\github\KonColle\KC\Logs"

    writer = SummaryWriter(log_dir=log_dir)

    # 数据预处理
    train_transforms = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transforms = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 划分训练/验证集
    category_image_paths = get_category_image_paths(base_dir)
    if not category_image_paths:
        print("未找到类别图片，请检查路径。")
        return
    train_images, val_images = split_dataset(category_image_paths, test_size=0.2)
    train_ann = r"D:\Programming\Project\github\KonColle\Datasets\annotations\instances_train.json"
    val_ann   = r"D:\Programming\Project\github\KonColle\Datasets\annotations\instances_val.json"
    filter_coco_annotations(annotation_file, train_images, train_ann)
    filter_coco_annotations(annotation_file, val_images, val_ann)

    # 检测类别映射
    det_id2name = extract_detection_classes(train_ann)
    num_det_classes = len(det_id2name)
    det_names = list(det_id2name.values())
    print(f"检测类别数: {num_det_classes}")

    # 数据加载
    train_ds = CustomCocoDataset(root=base_dir, annotation_file=train_ann,
                                  transform=train_transforms, image_files=train_images)
    val_ds   = CustomCocoDataset(root=base_dir, annotation_file=val_ann,
                                  transform=val_transforms,   image_files=val_images)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, collate_fn=custom_collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=4, collate_fn=custom_collate_fn)

    # 构建 YOLOv8 检测模型
    yolo_model_path = r"D:\Programming\Project\github\KonColle\KC\Models\YOLOv8\yolov8n.pt"
    yolo = YOLO(yolo_model_path)
    yolo.model.yaml['nc'] = num_det_classes
    yolo.model.yaml['names'] = det_names

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yolo.model.to(device)

    # 超参和损失
    default_hyp = {'box':7.5,'cls':0.5,'dfl':1.5,'pose':12.0,'kobj':1.0,'overlap_mask':True,'mask_ratio':4.0}
    if not hasattr(yolo.model, 'args') or yolo.model.args is None:
        yolo.model.args = AttrDict(default_hyp)
    else:
        yolo.model.args.update(default_hyp)
        yolo.model.args = AttrDict(yolo.model.args)

    loss_func = v8DetectionLoss(yolo.model)
    optimizer = AdamW(yolo.model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    scaler = GradScaler()

    best_wts = copy.deepcopy(yolo.model.state_dict())
    best_loss = float('inf')

    # 训练/验证循环
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for phase in ['train','val']:
            model = yolo.model
            loader = train_loader if phase=='train' else val_loader
            model.train() if phase=='train' else model.eval()
            running_loss = 0.0
            total = 0

            for imgs, det_targets in tqdm(loader, desc=phase):
                imgs = torch.stack([img.to(device) for img in imgs])
                # 构建 batch_targets
                tgt_list = []
                for i, t in enumerate(det_targets):
                    boxes = t['bboxes'].to(device)
                    labels= t['cls'].to(device).unsqueeze(1)
                    idxs  = torch.full((labels.size(0),1),i,device=device,dtype=torch.long)
                    tgt_list.append(torch.cat([idxs,labels,boxes],dim=1))
                if tgt_list:
                    tgt = torch.cat(tgt_list, dim=0)
                    batch_targets = {'batch_idx':tgt[:,0].long(),
                                     'cls':      tgt[:,1].long(),
                                     'bboxes':   tgt[:,2:6]}
                else:
                    batch_targets = {'batch_idx':torch.tensor([],dtype=torch.long,device=device),
                                     'cls':      torch.tensor([],dtype=torch.long,device=device),
                                     'bboxes':   torch.tensor([],dtype=torch.float32,device=device)}

                optimizer.zero_grad()
                with autocast(device.type):
                    preds = model(imgs)
                    det_loss,_ = loss_func(preds, batch_targets)
                if phase=='train':
                    scaler.scale(det_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                bs = imgs.size(0)
                running_loss += det_loss.item()*bs
                total += bs

            epoch_loss = running_loss/total if total>0 else 0.0
            print(f"{phase} Loss: {epoch_loss:.4f}")

            if phase=='val':
                scheduler.step(epoch_loss)
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_wts = copy.deepcopy(model.state_dict())
                    torch.save({'model':model,'detection_class_names':det_names}, model_save_path)
                    print(f"Best model saved to {model_save_path}")

    # 保存最终模型
    yolo.model.load_state_dict(best_wts)
    torch.save({'model':yolo.model,'detection_class_names':det_names}, model_save_path)
    print(f"Final model saved to {model_save_path}")
    writer.close()

if __name__ == '__main__':
    main()
