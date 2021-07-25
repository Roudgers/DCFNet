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
[VSOD Training dataset](https://pan.baidu.com/s/146uJcVmKsQhCZB708cb1kA). Code: oip1

## VSOD Testing dataset
[VSOD Testing dataset](https://pan.baidu.com/s/1z4uh5KIFIcMslPZMSKeyZw). Code: oip1

# Train/Test
## Testing
+ Firstly, you need to download the 'VSOD Testing Dataset' and the pretrained checkpoint we provided ([video_current_best_model.pth](https://pan.baidu.com/s/1qhyEke_1IwO5gVAkepHp0g). Code: oip1). 
+ Secondly, you need to set dataset path and checkpoint name correctly, and set the param '--split' as "test" or 'val' in inference.py for generating saliency results. 
+ Finally, you can evaluate the saliency results by using the widely-used tool provided by [DAVSOD](https://github.com/DengPingFan/DAVSOD).
+ Or you can directly download the results we provided ([DCFNet results](https://pan.baidu.com/s/1vAI28-SRTYRqn5JdQLpkjQ). Code: oip1).

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

# Acknowledge
Thanks to the previous helpful works:
+ [RCRNet](https://github.com/Kinpzz/RCRNet-Pytorch): 'Semi-Supervised Video Salient Object Detection Using Pseudo-Labels' by Pengxiang Yan, et al.
+ [SSAV](https://github.com/DengPingFan/DAVSOD): 'Shifting More Attention to Video Salient Object Detection' by Deng-Ping Fan, et al.

