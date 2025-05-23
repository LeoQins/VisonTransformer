
# @Brief: 数据加载模块，包含图像数据增强、加载以及数据集包装

from torchvision import transforms, datasets
import os
import glob
import torch
import cv2 as cv
import numpy as np
from PIL import Image

# 定义数据增强和预处理的转换操作
# 根据数据类型（训练或验证）选择不同的预处理方法
data_transform = {
    "train": transforms.Compose([
        # 随机调整图像尺寸并裁剪到224x224
        transforms.RandomResizedCrop(224),
        # 随机水平翻转图像，用于数据增强
        transforms.RandomHorizontalFlip(),
        # 将图像数据转换为Tensor格式
        transforms.ToTensor(),
        # 对图像进行归一化处理，均值和标准差均为0.5
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    "val": transforms.Compose([
        # 调整图像尺寸为256（保持长宽比）
        transforms.Resize(256),
        # 从图像中心裁剪出224x224的区域
        transforms.CenterCrop(224),
        # 将图像数据转换为Tensor格式
        transforms.ToTensor(),
        # 对图像进行归一化处理，均值和标准差均为0.5
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
}

# 定义一个基于内存中图像和标签的数据集类
class ImageLoader(torch.utils.data.Dataset):
    # 初始化时传入图像和标签的列表，以及是否采用数据增强
    def __init__(self, image_label, aug=False):
        self.image_label = image_label  # 图像和标签数据，列表形式
        self.aug = aug  # 是否进行数据增强标识

    # 定义获取索引处数据的方法，供DataLoader调用
    def __getitem__(self, index):
        # 从图像标签列表中获取第index个元素（图像，标签）
        image, label = self.image_label[index]

        # 根据是否启用数据增强选择相应的预处理转换
        if self.aug:
            # 如果启用数据增强，则使用训练时的转换
            image = data_transform["train"](image)
        else:
            # 否则使用验证时的转换
            image = data_transform["val"](image)

        # 返回处理后的图像及其标签
        return image, label

    # 返回数据集中样本的数量
    def __len__(self):
        return len(self.image_label)

# 定义一个基于图像路径的数据集类，用于从磁盘动态加载图像
class PathLoader(torch.utils.data.Dataset):
    # 初始化时传入图像路径和标签的列表，以及是否采用数据增强
    def __init__(self, image_label_path, aug=False):
        self.image_label_path = image_label_path  # 图像路径及标签的列表
        self.aug = aug  # 是否启用数据增强标识

    # 定义获取索引处数据的方法
    def __getitem__(self, index):
        # 从列表中获取第index个元素，包括图像路径和标签
        image_path, label = self.image_label_path[index]
        # 使用OpenCV读取图像文件
        image = cv.imread(image_path)
        # 将BGR格式转换为RGB格式（PIL库默认使用RGB）
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        # 将NumPy数组转换为PIL图像对象
        image = Image.fromarray(image)

        # 根据是否启用数据增强选择相应的预处理转换
        if self.aug:
            # 使用训练时的转换
            image = data_transform["train"](image)
        else:
            # 使用验证时的转换
            image = data_transform["val"](image)

        # 返回处理后的图像及其标签
        return image, label

    # 返回数据集中样本的数量
    def __len__(self):
        return len(self.image_label_path)

# 定义一个辅助函数，用于获得基于datasets.ImageFolder的数据加载器和数据集对象
def get_data_loader(data_dir, batch_size, num_workers, aug=False):
    # 使用ImageFolder读取数据，同时根据aug标识选择对应的预处理方式
    dataset = datasets.ImageFolder(root=data_dir, transform=data_transform["train" if aug else "val"])
    # 构建DataLoader对象，指定批量大小、是否打乱数据、以及加载线程数
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=batch_size, shuffle=aug,
                                         num_workers=num_workers)

    # 返回DataLoader和数据集对象以便后续使用或检查数据类别
    return loader, dataset
