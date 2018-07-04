# Scene Classification

This repository is to do scene classification by fine-tuning ResNet-152 with AI Challenger 2017 Scene Classification Dataset.


## Dependencies

- [NumPy](http://docs.scipy.org/doc/numpy-1.10.1/user/install.html)
- [Tensorflow](https://www.tensorflow.org/versions/r0.8/get_started/os_setup.html)
- [Keras](https://keras.io/#installation)
- [OpenCV](https://opencv-python-tutroals.readthedocs.io/en/latest/)

## Dataset

We use the Scene Classification Dataset from AI Challenger 2017, which contains 60,999 images of 80 classes of scenes. The data is split into 53,879 training images and 7,120 testing images.

 ![image](https://github.com/foamliu/Scene-Classification/raw/master/images/dataset.png)

You can get it from [Scene Classification Dataset](https://challenger.ai/datasets/scene):


## ImageNet Pretrained Models

Download [ResNet-152](https://drive.google.com/file/d/0Byy2AcGyEVxfeXExMzNNOHpEODg/view?usp=sharing) into models folder.

## Usage

### Data Pre-processing
Extract 60,999 training images, and split them (53,879 for training, 7,120 for validation):
```bash
$ python pre-process.py
```

### Train
```bash
$ python train.py
```

If you want to visualize during training, run in your terminal:
```bash
$ tensorboard --logdir path_to_current_dir/logs
```

### Demo
Download [pre-trained model](https://github.com/foamliu/Scene-Classification/releases/download/v1.0/model.85-0.7657.hdf5) into "models" folder then run:

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


### Analysis

```bash
$ python demo.py
```

Test_A | Test_B |
|---|---|
|0.9051136363636364|0|