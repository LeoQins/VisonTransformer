
# @Brief: 主程序，用于加载图像、模型，并执行预测


import os
import torch  # 引入PyTorch，用于深度学习相关操作
from PIL import Image  # 引入PIL库，用于图像处理

from dataloader import data_transform  # 导入数据预处理流程
from utils import create_model, model_parallel  # 导入模型创建和模型并行处理的工具函数
from config import args  # 导入配置参数


def main():
    # 根据是否有可用的CUDA设备决定使用GPU或CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 指定待预测图像的路径
    # image_path = "/home/rhdai/workspace/code/image_classification/dataset/daisy.jpg"
    image_path = "./dataset/daisy.jpg"  # 注意：请根据实际情况修改图像路径
    # 确认图像文件存在，否则抛出异常
    assert os.path.exists(image_path), "file: '{}' dose not exist.".format(image_path)

    # 打开图像文件
    image = Image.open(image_path)

    # 对图像进行预处理，预处理流程根据验证集设置(data_transform["val"])确定
    image = data_transform["val"](image)
    # 扩展图像的batch维度，转换为[N, C, H, W]格式，以适应模型输入要求
    image = torch.unsqueeze(image, dim=0)

    # 创建深度学习模型，参数通过配置文件(args)传入
    model = create_model(args)
    # 对模型应用并行化处理，确保模型在正确设备上运行
    model = model_parallel(args, model).to(device)

    # 指定预训练模型权重路径（请根据实际情况选择权重）
    # model_weight_path = "{}/weights/epoch=20_val_acc=0.9643.pth".format(args.summary_dir)
    model_weight_path = "{}/weights/epoch=3_val_acc=0.9396.pth".format(args.summary_dir)  # 加载预训练权重
    # 加载模型权重到模型中，并确保权重加载于当前设备上（GPU或CPU）
    model.load_state_dict(torch.load(model_weight_path, map_location=device))

    # 设置模型为评估模式，停用训练时特有的一些机制（如Dropout、BatchNorm的动态更新）
    model.eval()
    # 在关闭梯度计算的环境下进行预测，以节省内存和加快预测速度
    with torch.no_grad():
        # 将预处理后的数据送入模型进行预测，获取输出结果并去除多余的维度
        output = torch.squeeze(model(image.to(device))).cpu()
        # 对输出结果使用Softmax函数进行归一化，得到各类别的概率
        predict = torch.softmax(output, dim=0)
        # 通过寻找概率最大的索引，确定模型预测的类别
        index = torch.argmax(predict).numpy()

    # 打印预测的类别及其对应的概率
    print("prediction: {}   prob: {:.3}\n".format(args.label_name[index],
                                                predict[index].numpy()))
    # 遍历所有类别，打印每个类别的概率信息
    for i in range(len(predict)):
        print("class: {}   prob: {:.3}".format(args.label_name[i],
                                               predict[i].numpy()))


# 若本文件作为主程序执行，则调用main函数启动程序
if __name__ == '__main__':
    main()
