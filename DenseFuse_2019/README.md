# DenseFuse
* Densefuse 用于红外可见光图像融合。

* Densefuse 的 PyTorch 实现。fusion strategy 只用了最简单的addition。

* 论文地址：
  - [arXiv](https://arxiv.org/abs/1804.08361)

* 参考项目：
  - [DenseFuse官方代码](https://github.com/hli1221/imagefusion_densefuse)
  - [DenseFuse官方代码Pytorch版](https://github.com/hli1221/imagefusion_densefuse) 代码比较老
  - [DenseFuse-Refactoring-of-PyTorch](https://github.com/LGNWJQ/DenseFuse-Refactoring-of-PyTorch/tree/main) 主要学习了这里的代码
  - [bsun0802 / DenseFuse-pytorch](https://github.com/bsun0802/DenseFuse-pytorch) 这一篇看的比较少。在model.py中把fusion layer也写了进去，后续想完整复现融合策略可以参考这个的思路

* 百度网盘：
  - 链接：https://pan.baidu.com/s/1YlnmGa1jfCXt1YITPTV8ag?pwd=scp0 
  - 提取码：scp0

## 文件结构
```shell
├─fusion_test_data              # 用于测试的不同图片
│  ├─Road          	  	# Gray  可见光+红外
│  └─Tno           		# Gray  可见光+红外
│ 
├─fusion_result     # run_fusion.py 的运行结果。使用训练好的权重对fusion_test_data内图像融合结果 
│  ├─pair           # 单对图像融合结果
│  ├─Road_fusion
│  └─TNO_fusion
|
├─models                        # 网络模型
│  └─DenseFuse
│ 
├─runs              # run_train.py 的运行结果
│  └─train_07-15_16-28
│     ├─checkpoints # 模型权重
│     └─logs        # 用于存储训练过程中产生的Tensorboard文件
|
├─utils      	                # 调用的功能函数
│  ├─util_dataset.py            # 构建数据集
│  ├─util_device.py        	# 运行设备 
│  ├─util_fusion.py             # 模型推理
│  ├─util_loss.py            	# 结构误差损失函数
│  ├─util_train.py            	# 训练用相关函数
│  └─utils.py                   # 其他功能函数
│ 
├─configs.py 	    # 模型训练超参数
│ 
├─run_fusion.py   # 该文件使用训练好的权重将test_data内的测试图像进行融合
│ 
└─run_train.py      # 该文件用于训练模型

```



## 使用说明

### Trainng

#### 从零开始训练

* 打开configs.py对训练参数进行设置：
* 参数说明：

| 参数名              | 说明                                                                              |
|------------------|---------------------------------------------------------------------------------|
| image_path       | 用于训练的数据集的路径                                                                     |
| gray             | 为`True`时会进入灰度图训练模式，生成的权重用于对单通道灰度图的融合; 为`False`时会进入彩色RGB图训练模式，生成的权重用于对三通道彩色图的融合; |
| train_num        | `MSCOCO/train2017`数据集包含**118,287**张图像，设置该参数来确定用于训练的图像的数量                        |
| resume_path      | 默认为None，设置为已经训练好的**权重文件路径**时可对该权重进行继续训练，注意选择的权重要与**gray**参数相匹配                  |
| device           | 模型训练设备 cpu or gpu                                                               |
| batch_size       | 批量大小                                                                            |
| num_workers      | 加载数据集时使用的CPU工作进程数量，为0表示仅使用主进程，（在Win10下建议设为0，否则可能报错。Win11下可以根据你的CPU线程数量进行设置来加速数据集加载） |
| learning_rate    | 训练初始学习率                                                                            |
| num_epochs       | 训练轮数                                                                               |

* 设置完成参数后，运行**run_train.py**即可开始训练：

```python
    # 数据集相关参数
    parser.add_argument('--image_path', default=r'E:/project/Image_Fusion/DATA/COCO/train2017', type=str, help='数据集路径')
    parser.add_argument('--gray', default=True, type=bool, help='是否使用灰度模式')
    parser.add_argument('--train_num', default=70000, type=int, help='用于训练的图像数量')
    # 训练相关参数
    parser.add_argument('--device', type=str, default=device_on(), help='训练设备')
    parser.add_argument('--batch_size', type=int, default=2, help='input batch size, default=IR_images')
    parser.add_argument('--num_epochs', type=int, default=4, help='number of epochs to train for, default=4')
    parser.add_argument('--lr', type=float, default=1e-4, help='select the learning rate, default=1e-4')
    parser.add_argument('--resume_path', default=None, type=str, help='导入已训练好的模型路径')
    parser.add_argument('--num_workers', type=int, default=0, help='载入数据集所调用的cpu线程数')
    # 打印输出
    parser.add_argument('--output', action='store_true', default=True, help="shows output")
```

* 你可以在运行窗口看到类似的如下信息：

```
==================模型超参数==================
----------数据集相关参数----------
image_path: /home/COCO2017/train2017/
gray_images: True
train_num: 70000
----------训练相关参数----------
device: cuda
batch_size: 2
num_workers: 4
num_epochs: 4
learning rate : 0.0001
==================模型超参数==================
训练数据载入完成...
设备就绪...
Tensorboard 构建完成，进入路径：./runs/train_07-15_16-28/logs_Gray_epoch=4
然后使用该指令查看训练过程：tensorboard --logdir=./
测试数据载入完成...
initialize network with normal type
网络模型及优化器构建完成...
Epoch [1/4]: 100%|█| 35000/35000 [13:57<00:00, 41.78it/s, learning_rate=0.0001, pixel_loss=0.000146, ssim_lo
Epoch [2/4]: 100%|█| 35000/35000 [14:04<00:00, 41.43it/s, learning_rate=9e-5, pixel_loss=0.000298, ssim_loss
Epoch [3/4]: 100%|█| 35000/35000 [14:28<00:00, 40.29it/s, learning_rate=8.1e-5, pixel_loss=7.85e-5, ssim_los
Epoch [4/4]: 100%|█| 35000/35000 [14:33<00:00, 40.06it/s, learning_rate=7.29e-5, pixel_loss=2.43e-5, ssim_lo
Finished Training
训练耗时： 3425.805274248123
Best val loss: 0.000230

```

* Tensorboard查看训练细节：
  * **logs**文件夹下保存Tensorboard文件
  * 进入对于文件夹后使用该指令查看训练过程：`tensorboard --logdir=./`
  * 在浏览器打开生成的链接即可查看训练细节

#### 使用我提供的权重继续训练

* 打开args_fusion.py对训练参数进行设置
* 首先确定训练模式（Gray or RGB）
* 修改**resume_path**的默认值为已经训练过的权重文件路径

* 运行**run_train.py**即可运行



### Fuse Image

* 打开**run_fusion.py**文件，调整**defaults**参数
  * 确定融合模式（Gray or RGB）
  * 确定原图像路径和权重路径
  * 确定保存路径
* 运行**run_fusion.py**
* 你可以在运行窗口看到如下信息：

```shell
runs/train_07-15_16-28/checkpoints/epoch002-loss0.000.pth model loaded.
载入数据...
开始融合...
输出路径：fusion_result\Road_fusion/fusion_1.png
输出路径：fusion_result\Road_fusion/fusion_2.png
......
输出路径：fusion_result\Tno_fusion/fusion_14.png
输出路径：fusion_result\Tno_fusion/fusion_15.png
```











