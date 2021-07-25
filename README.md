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

# VSOD Training and Testing Datasets

## VSOD Training dataset
[Download Link](https://pan.baidu.com/s/1rduZEEo3HRq5HqQeXxuX-A). Code: nx8x

## VSOD Testing dataset
[Download Link](https://pan.baidu.com/s/1qkLkXpo8QEBqT_2nELPx1A). Code: qqsf

# Train/Test
## test
+ Firstly, you need to download the 'VSOD Testing Dataset' and the pretrained checpoint we provided ([Baidu Pan](https://pan.baidu.com/s/1xPH1AzInc1JAMq4Vq7UxGg). Code: d2o0). 
+ Then, you need to set dataset path and checkpoint name correctly. and set the param '--split' as "test" or 'val' in inference.py. 

```shell
python inference.py
```
## training
+ Firstly, you need to download the 'VSOD Training Dataset' and modify your path of training dataset
+ Secondly, you can pretrain the ImageModel for DCFNet following the training settings in the paper
+ Thirdly, you can train the VideoModel for DCFNet

```shell
python train.py
```

# Contact Us
If you have any questions, please contact us (1605721375@mail.dlut.edu.cn; dilemma@mail.dlut.edu.cn).

# Acknowledgement


