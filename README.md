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
[Download Link](https://pan.baidu.com/s/146uJcVmKsQhCZB708cb1kA). Code: oip1

## VSOD Testing dataset
[Download Link](https://pan.baidu.com/s/1z4uh5KIFIcMslPZMSKeyZw). Code: oip1

# Train/Test
## Testing
+ Firstly, you need to download the 'VSOD Testing Dataset' and the pretrained checpoint we provided ([Baidu Pan](https://pan.baidu.com/s/1qhyEke_1IwO5gVAkepHp0g). Code: oip1). 
+ Secondly, you need to set dataset path and checkpoint name correctly, and set the param '--split' as "test" or 'val' in inference.py for generating saliency results. 
+ Finally, you can evaluate the saliency results by using the widely-used tool 

```shell
python inference.py
```
## Training
+ Firstly, you need to download the 'VSOD Training Dataset' and modify your path of training dataset
+ Secondly, you can pretrain the ImageModel for DCFNet following the training settings in the paper
+ Finally, you can train the VideoModel for DCFNet

```shell
python train.py
```

# Contact Us
If you have any questions, please contact us (1605721375@mail.dlut.edu.cn; dilemma@mail.dlut.edu.cn).

# Acknowledgement


