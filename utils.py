# -*- coding: utf-8 -*-
# @File : utils.py
# @Author: Runist
# @Time : 2021/12/14 16:27
# @Software: PyCharm
# @Brief: 其他函数


import os
import torch
import shutil
import numpy as np
from torch import nn
from model import (vit_base_patch16_224_in21k,
                   vit_base_patch32_224_in21k,
                   vit_large_patch16_224_in21k,
                   vit_large_patch32_224_in21k,
                   vit_huge_patch14_224_in21k)#the last one too largre


def set_seed(seed):
    """
    设置随机种子
    Args:
        seed: 随机种子

    Returns: None

    """

    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)    # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def create_model(args):

    if args.model == "vit_base_patch16_224":
        model = vit_base_patch16_224_in21k(args.num_classes, has_logits=False)
    elif args.model == "vit_base_patch32_224":
        model = vit_base_patch32_224_in21k(args.num_classes, has_logits=False)
    elif args.model == "vit_large_patch16_224":
        model = vit_large_patch16_224_in21k(args.num_classes, has_logits=False)
    elif args.model == "vit_large_patch32_224":
        model = vit_large_patch32_224_in21k(args.num_classes, has_logits=False)
    elif args.model == "vit_huge_patch14_224":
        model = vit_huge_patch14_224_in21k(args.num_classes, has_logits=False)#too large
    else:
        raise Exception("Can't find any model name call {}".format(args.model))

    return model


def model_parallel(args, model):
    available_gpus = torch.cuda.device_count()
    gpu_list = [int(i) for i in args.gpu.split(',') if int(i) < available_gpus]
    if not gpu_list:
        raise ValueError("未找到有效的 GPU，系统中仅有 {} 块 GPU。".format(available_gpus))
    model = nn.DataParallel(model, device_ids=gpu_list)
    return model

def remove_dir_and_create_dir(dir_name):
    """
    清除原有的文件夹，并且创建对应的文件目录
    Args:
        dir_name: 该文件夹的名字

    Returns: None

    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print(dir_name, "Creat OK")
    else:
        shutil.rmtree(dir_name)
        os.makedirs(dir_name)
        print(dir_name, "Creat OK")