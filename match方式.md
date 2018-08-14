# ssd 目标检测 match方式

简单叙述一下最近看ssd caffe 源码心得, 看代码的时候要理解输入, 在下面举例子. 
caffe ssd 的大体思路是  

  1. base net (进行特征提取的基础网络, 原本为VGG, 也可用resnet 网络等) , base net 进行特征提取, 

  2. 产生多个分支, feature map 1, feature map 2,  ...  他们是从 base net 之后在不同尺度下进行特征提取, 

  3. feature map 产生3个输出, `Loc 回归`, `conf 分类` , `prior box`, 

     Loc 不是预测的坐标值,  而是 预测值 对 prior box 的一种映射关系, Loc 回归的就是这种映射关系

     映射关系可以表示为, ![1530203017990](C:\Users\BigB\AppData\Local\Temp\1530203017990.png)

解释一下: 损失回归的是  l (预测框和 默认框的映射关系),  与 g^ (ground truth 与默认框的 映射关系)