# 场景分类

微调 DenseNet-121, 解决 AI Challenger 2017 场景分类问题。 


## 依赖

- [NumPy](http://docs.scipy.org/doc/numpy-1.10.1/user/install.html)
- [Tensorflow](https://www.tensorflow.org/versions/r0.8/get_started/os_setup.html)
- [Keras](https://keras.io/#installation)
- [OpenCV](https://opencv-python-tutroals.readthedocs.io/en/latest/)

## 数据集

我们使用AI Challenger 2017中的场景分类数据集，其中包含80,900种场景的60,999张图像。 数据分为53,879个训练图像和7,120个测试图像。

 ![image](https://github.com/foamliu/Scene-Classification/raw/master/images/dataset.png)

你可以从中得到它 [Scene Classification Dataset](https://challenger.ai/datasets/scene):


## ImageNet 预训练模型

下载 [ResNet-152](https://drive.google.com/file/d/0Byy2AcGyEVxfeXExMzNNOHpEODg/view?usp=sharing) 放在 models 目录。

## 用法

### 数据预处理
Extract 60,999 training images, and split them (53,879 for training, 7,120 for validation):
```bash
$ python pre-process.py
```

### 训练
```bash
$ python train.py
```

If you want to visualize during training, run in your terminal:
```bash
$ tensorboard --logdir path_to_current_dir/logs
```

### Demo
下载 [pre-trained model](https://github.com/foamliu/Scene-Classification/releases/download/v1.0/model.85-0.7657.hdf5) 放在 models 目录然后执行:

```bash
$ python demo.py
```

1 | 2 | 3 | 4 |
|---|---|---|---|
|![image](https://github.com/foamliu/Scene-Classification/raw/master/images/0_out.png)  | ![image](https://github.com/foamliu/Scene-Classification/raw/master/images/1_out.png) | ![image](https://github.com/foamliu/Scene-Classification/raw/master/images/2_out.png)| ![image](https://github.com/foamliu/Scene-Classification/raw/master/images/3_out.png) |
|![image](https://github.com/foamliu/Scene-Classification/raw/master/images/4_out.png)  | ![image](https://github.com/foamliu/Scene-Classification/raw/master/images/5_out.png) | ![image](https://github.com/foamliu/Scene-Classification/raw/master/images/6_out.png)| ![image](https://github.com/foamliu/Scene-Classification/raw/master/images/7_out.png) |
|![image](https://github.com/foamliu/Scene-Classification/raw/master/images/8_out.png)  | ![image](https://github.com/foamliu/Scene-Classification/raw/master/images/9_out.png) |![image](https://github.com/foamliu/Scene-Classification/raw/master/images/10_out.png) | ![image](https://github.com/foamliu/Scene-Classification/raw/master/images/11_out.png)|
|![image](https://github.com/foamliu/Scene-Classification/raw/master/images/12_out.png)  | ![image](https://github.com/foamliu/Scene-Classification/raw/master/images/13_out.png) |![image](https://github.com/foamliu/Scene-Classification/raw/master/images/14_out.png)| ![image](https://github.com/foamliu/Scene-Classification/raw/master/images/15_out.png)|
|![image](https://github.com/foamliu/Scene-Classification/raw/master/images/16_out.png) | ![image](https://github.com/foamliu/Scene-Classification/raw/master/images/17_out.png) | ![image](https://github.com/foamliu/Scene-Classification/raw/master/images/18_out.png) | ![image](https://github.com/foamliu/Scene-Classification/raw/master/images/19_out.png) |


### 结果分析

```bash
$ python analyze.py
```

| |Test A|Test B|
|---|---|---|
|Top3准确度|0.9051136363636364|0|