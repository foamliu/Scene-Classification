# 场景分类

微调 Inception-ResNet-V2, 解决 AI Challenger 2017 场景分类问题。


## 依赖

- [NumPy](http://docs.scipy.org/doc/numpy-1.10.1/user/install.html)
- [Tensorflow](https://www.tensorflow.org/versions/r0.8/get_started/os_setup.html)
- [Keras](https://keras.io/#installation)
- [OpenCV](https://opencv-python-tutroals.readthedocs.io/en/latest/)

## 数据集

我们使用AI Challenger 2017中的场景分类数据集，其中包含80,900种场景的60,999张图像。 数据分为53,879个训练图像和7,120个测试图像。

 ![image](https://github.com/foamliu/Scene-Classification/raw/master/images/dataset.png)

你可以从中得到它 [Scene Classification Dataset](https://challenger.ai/datasets/scene):

### 性能
用14118张测试图片计算平均准确率(mAP)，结果如下：

| |Test A|Test B|
|---|---|---|
|图片数|7040|7078|
|Top3准确度|0.94346|0.91212|

## 用法

### 数据预处理
提取60,999个训练图像，并将它们分开（53,879个用于训练，7,120个用于验证）：
```bash
$ python pre-process.py
```

### 训练
```bash
$ python train.py
```

如果想在培训期间进行可视化，请在终端中运行：
```bash
$ tensorboard --logdir path_to_current_dir/logs
```

### Demo
下载 [pre-trained model](https://github.com/foamliu/Scene-Classification/releases/download/v1.0/model.11-0.6262.hdf5) 放在 models 目录然后执行:

```bash
$ python demo.py
```

1 | 2 | 3 | 4 |
|---|---|---|---|
|![image](https://github.com/foamliu/Scene-Classification/raw/master/images/0_out.png)  | ![image](https://github.com/foamliu/Scene-Classification/raw/master/images/1_out.png) | ![image](https://github.com/foamliu/Scene-Classification/raw/master/images/2_out.png)| ![image](https://github.com/foamliu/Scene-Classification/raw/master/images/3_out.png) |
|教室, prob: 0.751|修理店, prob: 0.4876|沙漠, prob: 0.9402|酒吧, prob: 0.8236|
|![image](https://github.com/foamliu/Scene-Classification/raw/master/images/4_out.png)  | ![image](https://github.com/foamliu/Scene-Classification/raw/master/images/5_out.png) | ![image](https://github.com/foamliu/Scene-Classification/raw/master/images/6_out.png)| ![image](https://github.com/foamliu/Scene-Classification/raw/master/images/7_out.png) |
|宫殿, prob: 0.6837|博物馆, prob: 0.6911|住宅, prob: 0.5338|会议室, prob: 0.9461|
|![image](https://github.com/foamliu/Scene-Classification/raw/master/images/8_out.png)  | ![image](https://github.com/foamliu/Scene-Classification/raw/master/images/9_out.png) |![image](https://github.com/foamliu/Scene-Classification/raw/master/images/10_out.png) | ![image](https://github.com/foamliu/Scene-Classification/raw/master/images/11_out.png)|
|集市, prob: 0.9636|桥, prob: 0.571|航站楼, prob: 0.9362|游乐场, prob: 0.5429|
|![image](https://github.com/foamliu/Scene-Classification/raw/master/images/12_out.png)  | ![image](https://github.com/foamliu/Scene-Classification/raw/master/images/13_out.png) |![image](https://github.com/foamliu/Scene-Classification/raw/master/images/14_out.png)| ![image](https://github.com/foamliu/Scene-Classification/raw/master/images/15_out.png)|
|保龄球馆, prob: 0.9995|漂流, prob: 0.998|水族馆, prob: 0.9898|停机坪, prob: 0.9965|
|![image](https://github.com/foamliu/Scene-Classification/raw/master/images/16_out.png) | ![image](https://github.com/foamliu/Scene-Classification/raw/master/images/17_out.png) | ![image](https://github.com/foamliu/Scene-Classification/raw/master/images/18_out.png) | ![image](https://github.com/foamliu/Scene-Classification/raw/master/images/19_out.png) |
|跑马场, prob: 0.9966|实验室, prob: 0.8698|滑雪场, prob: 0.8024|会议室, prob: 0.6975|


### 性能评估
```bash
$ python evaluate.py
```
