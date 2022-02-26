# animalfruit
# 作业四：撰写项目README并完成开源
## 一、项目背景介绍
近年来，随着人工智能机器视觉这一分支技术的不断发展突破，CV+AI开始不断渗透至生活乃至工作等各类场景之中。而在人脸识别应用逐步普及之后，借助CV+AI对动物进行识别，已成为了当下多数企业看好的一个全新发力点。

之所以机器视觉开始进入动物识别领域，其原因还要先从技术本身说起。

### 1.1 识别动物与识别人之间的差距

动物识别这条路之所以行得通，其原因在于动物与人类相仿，都有能够进行识别的生物特性。最典型的代表便是狗鼻子上的鼻纹，与人类指纹相似，这是一种与生俱来且独一无二的生物特性，即便同一胎产下的狗，鼻纹也存在很大差异。

当然，此前的AI养猪借鉴的也是同一原理。借助每头猪不同的面部特征（两眼间的距离、嘴巴的位置、头骨的宽度），配合机器视觉结构化出的数据以及人工智能在数据处理方面的优势，才让这一解决方案有了落地的可能性。

这也是为何，机器视觉+人工智能（也就是CV+AI）近年来一直深耕养殖业的原因。毕竟相比种植产业来说，养殖业因素更为可控。而种植业即使在同一区域内，还要考虑气候波动等各项因素。

回归VC+AI这样话题来看，此前比特网在与一位业界人士沟通时，其表示动物脸识别其实跟人脸识别的原理相通。本质上来说采用的技术是相似的，只不过训练选用的样本有所差异而已。现阶段CV+AI炒的很热，其实最终并不在于企业算法有多高深，核心还是在于数据。谁能够设计出一个讨巧低成本，且能够源源不断产生数据的一个模式，并借助这一模式来反哺到数据之上，进行算法方面的优化，谁才能成功卡位这一市场当中。

而从另一个角度来看的话，以往在养猪时，若想对个体进行识别，通常采用的是射频识别（RFID）方式，也就是在猪耳部位以打孔的方式来添加一个标签。但此类方式的弊端在于，识别范围受限不能同时读取多类标签，且准确率也差强人意。同时这种方式在打孔时，亦会给牲畜带来极大的痛楚，因此在当下传统方式也因不人道而屡遭批评。

### 1.2 CV+AI动物识别背后的几点思考

CV+AI近年来之所以能够不断渗透至市场当中，一方面得益于数字化转型的带动，整个大环境发展趋势，也在朝着以科技为主导的规模化、精细化和智能化的方向上走。

纵观全球农业发展趋势，美国注重农业科技创新，始终保持着农业技术的全球竞争优势。比如利用机器学习，减少产量损失与劳动力成本，亦或是通过对物联网基础设施数据的收集，结合算法将其变为可视化的指导性数据，给出最佳的种植及市场方案。

除农业市场外，其他市场也给了CV+AI很大的渗透空间。以宠物市场为例，根据国家统计局的数据显示，2010年至2016年期间，中国宠物行业的年增长率接近50%。

另一项来自中国宠物产业白皮书也指出，2018年国内已拥有将近3400万狗主和2260万猫主，宠物产业的数量在去年达到了1710亿元。

而智研咨询发布的数据则显示，2019 年中国城镇宠物（犬猫）消费市场规模首次突破 2000 亿大关，达到 2024 亿元，比 2018 年增长 18.5%，近 10 年 CAGR为 34.55%。

因此在宠物市场持续走高的背后，问题也接踵而至。一方面爱宠走失后很大几率无法被找回，室外张贴寻狗启示这种笨方式，不仅覆盖的范围有限，更会对环境带来负面影响，此外一些细节特征除饲养者外鲜有有外人能够辨别。

再者，与日俱增的宠物数量，亦对城市治理提出了全新挑战，如何以可行性方案来管控不文明养犬行为，更成为了留给城市治理过程中不得不正视的一个课题。
## 二、数据介绍
这里用到的数据集是第十六届智能车视觉AI组组委会提供的数据集

![image.png](attachment:5cc95ecb-b7ff-4d43-bc5e-363dc5453441.png)![image.png](attachment:191f34d5-b569-4ea8-80e7-4549d1fced53.png)

### 2.1 继承paddle.io.IterableDataset类
```
import os
import random

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import paddle
class myreader(paddle.io.Dataset):


if _
_name__ == '__main__':
	pass
```

### 2.2 实现构造函数，定义数据读取方式，划分训练和测试数据集
将数据集按照数字索引的方式命名，要处理的文件路径是这样的：

![image.png](attachment:4504e452-a49f-4bed-b8f5-5d9e27e75da0.png)
0~9是分别是5个动物和5个水果大类，里面分别有各类png图片100张左右。
用os.path.join()方法来拼接路径，遍历得到每一张图片的路径。
然后用os.path.splitext()来判段文件是否是.png后缀的，这个方法会将文件名和文件后缀分开，os.path.splitext(文件路径)[-1]所读取到的就是后缀。
然后用cv.imread来读取图片，输入参数有两个，一个是路径，一个是读取的色彩选择，可以选择bgr彩图还是灰度图，默认bgr彩图，注意这个顺序是bgr,而不是常见的rgb,如果读取完后直接用plt来显示的话，图像的色彩就会出错。
所以我们用cv.resize()来对图像的大小以及色彩进行修改，cv.resize(img, (64, 64))[…, (2, 1, 0)],这是指将图像放缩为64×64大小的rgb图片。（这里修改图像大小的步骤不能省去，否则数组形状不一，后面会报错或警告）
我们需要将整组图片分为训练集以及测试集，这两个集合不能有重叠部分。这里用train_ratio作为参数来指定训练集的占比，然后用train_ratio×10与图片索引相比来决定是训练集还是测试集，这样的话对于相同的一组图片以及相同的train_ratio来说其训练集和测试集完全没有重叠
## 三、模型介绍
飞桨框架支持两种组网方式，一种是Sequential组网，另一种是SubClass组网。Sequential 可以快速的完成组网。但是当我们想组建一些比较复杂的网络结构可能就需要SubClass组网了。我使用的是Sequential 来快速搭建一个简单的网络。
这个网络由卷积层，池化层，激活函数层和线性变换层构成。
![image.png](attachment:c930bed7-ac47-4bc7-a4a7-943d37f32c84.png)![image.png](attachment:37049c2b-2792-4ad9-95e1-49a9f655c240.png)
## 四、模型训练
先设置一下optimizer优化器，loss损失函数，metrics设置精度计算方式
```
model.prepare(optimizer=paddle.optimizer.Adam(parameters=model.parameters()),
	loss=paddle.nn.CrossEntropyLoss(),
	metrics=paddle.metric.Accuracy())
```
然后使用高层API train_batch（） 完成单个批次数据的训练操作。
```
epoch_num = 25   # 设置训练轮次
for epoch in range(epoch_num):
    for batch_id, batch_data in enumerate(train_loader):
        inputs = batch_data[0]
        labels = batch_data[1]
        out = model.train_batch([inputs], [labels])
        if batch_id % 2 == 0:
            print('epoch: {}, batch: {}, loss: {}, acc: {}'.format(epoch, batch_id, out[0][0], out[1]))
    
    eval_result = model.evaluate(test_dataset, verbose=1)
    print('测试',eval_result['loss'][0])
    print('测试acc',eval_result['acc'])
```
![image.png](attachment:dc010cf2-116e-4125-b2f2-b01f1cd4de5d.png)![image.png](attachment:ae8717a0-4947-4f01-93c4-963607bdb97e.png)
## 五、模型评估
```
# 用 model.eval_batch 在测试集一个批次的数据上进行验证
for batch_id, batch_data in enumerate(test_loader):
    inputs = batch_data[0]
    labels = batch_data[1]
    test_result = model.eval_batch([inputs],[labels])
    print("predict finished")
    print(test_result)

```
![image.png](attachment:d269adf9-bfc4-4e0e-90c4-c0705f8c63f6.png)![image.png](attachment:96c8ec54-c15f-4fb0-bd24-970ad6ada9dd.png)
## 六、总结与升华
项目识别率还不够高，之后再对模型和参数进行改进。
## 七、个人总结
我是ai小白，计算机系大三在读。
