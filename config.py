# -*- coding: utf-8 -*-
# 导入argparse模块，用于处理命令行参数
import argparse
# 导入os模块，用于环境变量操作
import os

# 创建命令行参数解析器
parser = argparse.ArgumentParser()

# 添加命令行参数：类别数量，默认值为5
parser.add_argument('--num_classes', type=int, default=5)
# 添加命令行参数：训练轮数，默认值为30
parser.add_argument('--epochs', type=int, default=30)
# 添加命令行参数：批量大小，默认值为32
parser.add_argument('--batch_size', type=int, default=32)
# 添加命令行参数：学习率，默认值为0.001
parser.add_argument('--lr', type=float, default=0.001)
# 添加命令行参数：学习率衰减因子，默认值为0.01
parser.add_argument('--lrf', type=float, default=0.01)

# 添加训练数据目录的命令行参数，默认路径为"./dataset/train/"，并添加帮助说明
parser.add_argument('--dataset_train_dir', type=str,
                    default="./dataset/train/",
                    help='The directory containing the train data.')
# 添加验证数据目录的命令行参数，默认路径为"./dataset/validation/"，并添加帮助说明
parser.add_argument('--dataset_val_dir', type=str,
                    default="./dataset/validation/",
                    help='The directory containing the val data.')
# 添加保存权重和tensorboard日志的目录参数，默认路径为"./summary/vit_base_patch16_224"，并添加帮助说明
parser.add_argument('--summary_dir', type=str, default="./summary/vit_base_patch16_224",
                    help='The directory of saving weights and tensorboard.')

# 添加预训练权重路径参数，默认值为'./pretrain_weights/vit_base_patch16_224_in21k.pth'
# 如果不想加载预训练权重，可以设置为空字符串
parser.add_argument('--weights', type=str, default='./pretrain_weights/vit_base_patch16_224_in21k.pth',
                    help='Initial weights path.')

# 添加是否冻结权重参数，使用布尔类型，默认设置为True表示冻结部分模型权重
parser.add_argument('--freeze_layers', type=bool, default=True)
# 添加选择显卡的命令行参数，默认使用0,1号gpu，同时提供帮助说明
parser.add_argument('--gpu', type=str, default='0,1', help='Select gpu device.')  # 默认使用0,1号gpu

# 添加模型名称参数，默认值为'vit_base_patch16_224'，提供帮助说明用于选择要训练的ViT模型
parser.add_argument('--model', type=str, default='vit_base_patch16_224',
                    help='The name of ViT model, Select one to train.')
# 添加类别名称参数，默认值为一个包含五个花卉名称的列表，提供帮助说明表示各类别的名称
parser.add_argument('--label_name', type=list, default=[
    "daisy",
    "dandelion",
    "roses",
    "sunflowers",
    "tulips"
], help='The name of class.')

# 解析命令行参数，将结果存储在args中
args = parser.parse_args()

# 根据传入的gpu参数设置CUDA可见设备，确保只使用指定的GPU
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
