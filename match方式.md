# ssd 目标检测 match方式

简单叙述一下最近看ssd caffe 源码心得,
caffe ssd 的大体思路是  

 	1. base net (进行特征提取的基础网络, 原本为VGG, 也可用resnet 网络等) , base net 进行特征提取, 
 	2. 产生多个分支, feature map 1, feature map 2,  ...  他们是从 base net 之后在不同尺度下进行