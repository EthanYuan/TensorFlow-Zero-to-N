# 经典CNN

## Alex（2012）

### 基本信息

- 错误率：16.4%；
- 6000万个参数；
- 5个卷积层，3个全连接层；

### 架构

- CONV1
- MAX POOL1
- NORM1
- CONV2
- MAX POOL2
- NORM2
- CONV3
- CONV4
- CONV5
- MAXPOOL3
- FC6
- FC7
- FC8

### 技术特点

- 首次使用ReLU；
- 使用Norm层（现在不常见了）；
- 大量数据扩展；
- 使用SGD Momentum 0.9；
- Dropout 0.5；
- 7 CNN ensemble；
- L2正则；

### 超参数

- batch size 128；
- learning rate 1e-2，手动减小当精度趋于平缓时；

## ZFNet（2013）

### 基本信息

- 错误率：11.7%
- 8层；

### 技术特点

- 基于AlexNet；
- CONV1：从（11x11，stride 4）变为（7x7 stride 2）
- CONV3,4,5：由384，384，256变为512，1024，512

## VGG（2014）

### 基本信息

- 错误率：7.3%
- 19层；


### 架构

- conv3-64
- conv3-64
- maxpool
- conv3-128
- conv3-128
- maxpool
- conv3-256
- conv3-256
- conv3-256
- conv3-256
- maxpool
- conv3-512
- conv3-512
- conv3-512
- conv3-512
- maxpool
- conv3-512
- conv3-512
- conv3-512
- conv3-512
- maxpool
- FC-4096
- FC-4096
- FC-1000
- softmax

### 技术特点

- 3x3 CONV stride 1，pad 1；
- 2x2 MAX POOL stride 2；

### 启示

- LRN层作用不大；
- 越深的网络效果越好；
- 1x1的卷积也是很有效的，但是没有3x3的卷积好，大一些的卷积核可以学习更大的空间特征；
- 3个3x3的卷积层串联的效果相当于1个7x7的卷积层，前者只有后者55%的参数量，更多的非线性变换；

## GoogleNet（2014）

### 基本信息

- 错误率：6.70%
- 500百万参数（参数量只有AlexNet 6000万的1/12）；
- 比AlexNet速度快2倍；
- 22层；

### 技术特点

- Inception module；
- 完全移除了全连接层，用全局平均池化层来取代，全连接层占据了VGG和AlexNet 90%的参数量；

### Inception Family

- Inception V2，提出了BN，可以去除Dropout、LRN，传统的DNN在每一层的输入的分布都在变化，导致训练变得困难，我们只能使用一个很小的学习率解决这个问题。
- Inception V3，非对称的卷积结构拆分，优化了Inception Module的结构；
- Inception V4，结合了微软的ResNet；

## ResNet（2015）

### 基本信息

- 2~3周，8 GPU机器来进行训练；
- 运行速度比VGG要快；

### 启示

- plain net
- residual
