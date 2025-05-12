# Vision Transformer

## Introduction

![ViT.png](https://s2.loli.net/2022/01/19/w3CyXNrhEeI7xOF.png)

Network for Vision Transformer. The pytorch version. 

If this works for you, please give me a star, this is very important to me.😊

## Quick start

1. Clone this repository

```shell
git clone https://github.com/LeoQins/VisonTransformer
```
2. Install torch_Vision_Transformer from source.

```shell
cd torch_Vision_Transformer
pip install -r requirements.txt
```

3. Modifying the [config.py](https://github.com/LeoQins/VisonTransformer/blob/main/config.py).
4. Download pretrain weights, the url in [utils.py](https://github.com/LeoQins/VisonTransformer/blob/main/utils.py).
5. Start train your model.

```shell
python train.py
```
6. Open tensorboard to watch loss, learning rate etc. You can also see training process and training process and validation prediction.

```shell
tensorboard --logdir "......\summary\vit_base_patch16_224\logs" #windows要求绝对路径，Linux可以相对路径
```
![tensorboard.png](https://s2.loli.net/2022/10/12/p7KtB1uXMkqvreN.png)

7. Get prediction of model.

```shell
python predict.py
```

## Train your dataset

You need to store your data set like this:

```shell
dataset
├── train
│   ├── daisy
│   ├── dandelion
│   ├── roses
│   ├── sunflowers
│   └── tulips
└── validation
    ├── daisy
    ├── dandelion
    ├── roses
    ├── sunflowers
    └── tulips
```

## other folders
```shell
workspace
├── pretrain_weights
│   ├── vit_base_patch16_224_in21k.pth
│   ├── vit_base_patch32_224_in21k
│   ├── vit_large_patch16_224_in21k.pth
│   └── vit_large_patch32_224_in21k.pth
└── summary
    ├── logs
    └── weights
```



## Reference

Appreciate the work from the following repositories:

- [WZMIAOMIAO](https://github.com/WZMIAOMIAO)/[vision_transformer](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_classification/vision_transformer)
- [Runist](https://github.com/Runist)/[torch_vison_transformer](https://github.com/Runist/torch_Vision_Transformer)




## License

Code and datasets are released for non-commercial and research purposes **only**. For commercial purposes, please contact the authors.

