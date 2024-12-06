# DeepFuse

---

### The re-implementation of ICCV 2017 DeepFuse paper idea

![](figure\Schematic%20diagram.png)

![](figure\framework.png)

This code is based on [K. Ram Prabhakar, V Sai Srikar, R. Venkatesh Babu. DeepFuse: A Deep Unsupervised Approach for Exposure Fusion with Extreme Exposure Image Pairs. ICCV2017, pp. 4714-4722](http://openaccess.thecvf.com/content_iccv_2017/html/Prabhakar_DeepFuse_A_Deep_ICCV_2017_paper.html)

---

## Description 描述
- **基础框架：** CNN
- **任务场景：** 用于多曝光图像融合，multi-exposure fusion (MEF)。
- **项目描述：** DeepFuse 的 PyTorch 实现。fusion strategy 论文中只用了最简单的addition。
- **论文地址：**
  - [readpaper.com](https://readpaper.com/home/) 
  - [K. Ram Prabhakar, V Sai Srikar, R. Venkatesh Babu. DeepFuse: A Deep Unsupervised Approach for Exposure Fusion with Extreme Exposure Image Pairs. ICCV2017, pp. 4714-4722](http://openaccess.thecvf.com/content_iccv_2017/html/Prabhakar_DeepFuse_A_Deep_ICCV_2017_paper.html)
- **参考项目：**
  - [SunnerLi/DeepFuse.pytorch](https://github.com/SunnerLi/DeepFuse.pytorch) 主要学习了这里的代码。
  - [sndnshr/DeepFuse](https://github.com/sndnshr/DeepFuse)
  - [thfylsty/DeepFuse](https://github.com/thfylsty/Classic-and-state-of-the-art-image-fusion-methods/tree/main/deepfuse)

- **百度网盘：**
  - 链接：[DeepFuse Link](https://pan.baidu.com/s/12cvVquASeokZ7xkvM7sHeA?pwd=scp0)
  - 提取码：scp0

---

## Idea 想法

In this code, for all conv layers, the filter size is 3*3. And this code is not a complete version for DeepFuse, we just implement one channel fusion method which use CNN network.

This code is not exactlly same with paper in ICCV2017. The aim of the training process is to reconstruct the input image by this network. The encoder(C1, C2) is used to extract image features and decoder(C3, C4, C5) is a reconstruct tool. The fusion strategy( Tensor addition) is only used in testing process.

We train this network using [Microsoft COCO dataset](http://msvocds.blob.core.windows.net/coco2014/train2014.zip)(T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollar, and C. L. Zitnick. Microsoft coco: Common objects in context. In ECCV, 2014. 3-5.) as input images which contains 80000 images and all resize to 256×256 and RGB images are transformed to gray ones.

---

## Structure 文件结构

```shell
├─data_test              # 用于测试的不同图片
│ 
├─data_result     # run_infer.py 的运行结果。使用训练好的权重对fusion_test_data内图像融合结果 
│  └─pair           # 单对图像融合结果
|
├─models                        # 网络模型
│  └─DeepFuse
│ 
├─runs              # run_train.py 的运行结果
│  └─train_03-03_16-00
│     ├─checkpoints # 模型权重
│     └─logs        # 用于存储训练过程中产生的Tensorboard文件
|
├─utils                          # 调用的功能函数
│  ├─util_dataset.py            # 构建数据集
│  ├─util_device.py            # 运行设备 
│  ├─util_fusion.py             # 模型推理
│  ├─util_loss.py                # 结构误差损失函数
│  ├─util_train.py                # 训练用相关函数
│  └─utils.py                   # 其他功能函数
│ 
├─configs.py         # 模型训练超参数
│ 
├─run_infer.py   # 该文件使用训练好的权重将test_data内的测试图像进行融合
│ 
└─run_train.py      # 该文件用于训练模型
```
---
## Usage 使用说明

### Trainng

#### 从零开始训练

* 打开configs.py对训练参数进行设置：
* 参数说明：

| 参数名           | 说明                                                                                  |
| ------------- | ----------------------------------------------------------------------------------- |
| image_path    | 用于训练的数据集的路径                                                                         |
| train_num     | 设置该参数来确定用于训练的图像的数量                                                                  |
| resume_path   | 默认为None，设置为已经训练好的**权重文件路径**时可对该权重进行继续训练，注意选择的权重要与**gray**参数相匹配                      |
| device        | 模型训练设备 cpu or gpu                                                                   |
| batch_size    | 批量大小                                                                                |
| num_workers   | 加载数据集时使用的CPU工作进程数量，为0表示仅使用主进程，（在Win10下建议设为0，否则可能报错。Win11下可以根据你的CPU线程数量进行设置来加速数据集加载） |
| learning_rate | 训练初始学习率                                                                             |
| num_epochs    | 训练轮数                                                                                |

* 设置完成参数后，运行**run_train.py**即可开始训练：

```python
    parser = argparse.ArgumentParser(description="模型参数设置")
    # 数据集相关参数
    parser.add_argument('--image_path', default=r'E:/project/Image_Fusion/DATA/MEF_DATASET', type=str, help='数据集路径')
    parser.add_argument('--train_num', default=200, type=int, help='用于训练的图像数量')
    # 训练相关参数
    parser.add_argument('--device', type=str, default=device_on(), help='训练设备')
    parser.add_argument('--batch_size', type=int, default=4, help='input batch size, default=4')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs to train for, default=10')
    parser.add_argument('--lr', type=float, default=1e-4, help='select the learning rate, default=1e-2')
    parser.add_argument('--num_workers', type=int, default=0, help='载入数据集所调用的cpu线程数')
    parser.add_argument('--resume_path', default=None, type=str, help='导入已训练好的模型路径')
    # 打印输出
    parser.add_argument('--output', action='store_true', default=True, help="shows output")
    # 使用parse_args()解析参数
    args = parser.parse_args()
```

* 你可以在运行窗口看到类似的如下信息：

```
==================模型超参数==================
----------数据集相关参数----------
image_path: E:/project/Image_Fusion/DATA/MEF_DATASET
train_num: 200
----------训练相关参数----------
device: cpu
batch_size: 4
num_epochs: 100
num_workers: 0
learning rate : 0.0001
resume_path: None
==================模型超参数==================
Statistic the over-exposure and under-exposure image list...: 100%|██████████| 200/200 [06:39<00:00,  2.00s/it]
  0%|          | 0/50 [00:00<?, ?it/s]训练数据载入完成...
设备就绪...
Tensorboard 构建完成，进入路径：./runs\train_07-22_14-01\logs
然后使用该指令查看训练过程：tensorboard --logdir=./
测试数据载入完成...
initialize network with normal type
网络模型及优化器构建完成...
Epoch [1/100]: 100%|██████████| 50/50 [00:03<00:00, 14.34it/s, learning_rate=0.0001, loss=0.393]
Epoch [2/100]: 100%|██████████| 50/50 [00:03<00:00, 13.14it/s, learning_rate=9e-5, loss=0.351]
Epoch [3/100]: 100%|██████████| 50/50 [00:04<00:00, 12.49it/s, learning_rate=8.1e-5, loss=0.23]
Epoch [4/100]: 100%|██████████| 50/50 [00:03<00:00, 13.88it/s, learning_rate=7.29e-5, loss=0.211]
Epoch [5/100]: 100%|██████████| 50/50 [00:03<00:00, 13.56it/s, learning_rate=6.56e-5, loss=0.18]
Epoch [6/100]: 100%|██████████| 50/50 [00:03<00:00, 13.84it/s, learning_rate=5.9e-5, loss=0.38]
......
Epoch [98/100]: 100%|██████████| 50/50 [00:04<00:00, 10.62it/s, learning_rate=3.64e-9, loss=0.282]
Epoch [99/100]: 100%|██████████| 50/50 [00:04<00:00, 10.45it/s, learning_rate=3.28e-9, loss=0.157]
Epoch [100/100]: 100%|██████████| 50/50 [00:04<00:00, 10.36it/s, learning_rate=2.95e-9, loss=0.263]
Finished Training
训练耗时： 419.81079602241516
Best loss: 0.212551
```

* Tensorboard查看训练细节：
  * **logs**文件夹下保存Tensorboard文件
  * 进入对于文件夹后使用该指令查看训练过程：`tensorboard --logdir=./`
  * 在浏览器打开生成的链接即可查看训练细节

#### 使用提供的权重继续训练

* 打开configs.py对训练参数进行设置
* 修改**resume_path**的默认值为已经训练过的权重文件路径
* 运行**run_train.py**即可运行

### Fuse Image

* 打开**run_fusion.py**文件，调整**defaults**参数
  * 确定原图像路径和权重路径
  * 确定保存路径
* 运行**run_fusion.py**
