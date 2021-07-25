# DCFNet
Accepted paper in ICCV2021:

Dynamic Context-Sensitive Filtering Network for Video Salient Object Detection

Miao Zhang, Jie Liu, Yifei Wang, [Yongri Piao](http://ice.dlut.edu.cn/yrpiao/), Shunyu Yao, Wei Ji, Jingjing Li, [Huchuan Lu](http://ice.dlut.edu.cn/lu/publications.html), [Zhongxuan Luo](zxluo@dlut.edu.cn).

# Prerequisites
+ Ubuntu 16
+ PyTorch 1.6.0
+ CUDA 10.1
+ Cudnn 7.5.0
+ Python 3.6

# Training and Testing Datasets

## Training dataset
[Download Link](https://pan.baidu.com/s/1rduZEEo3HRq5HqQeXxuX-A). Code: nx8x

## Testing dataset
[Download Link](https://pan.baidu.com/s/1qkLkXpo8QEBqT_2nELPx1A). Code: qqsf

# Train/Test
## test
Firstly, you need to download the 'Testing dataset' and the pretraind checpoint we provided ([Baidu Pan](https://pan.baidu.com/s/1xPH1AzInc1JAMq4Vq7UxGg). Code: d2o0). Then, you need to set dataset path and checkpoint name correctly. and set the param '--phase' as "test" and '--param' as 'True' in demo.py. 

```shell
python demo.py
```
## train
Once the train-augment dataset are prepared,you need to set dataset path and checkpoint name correctly. and set the param '--phase' as "train" and '--param' as 'True'(loading checkpoint) or 'False'(do not load checkpoint) in demo.py. 

```shell
python demo.py
```

# Contact Us
If you have any questions, please contact us (xiaofeisun@mail.dlut.edu.cn; 1605721375@mail.dlut.edu.cn).

# Acknowledgement


