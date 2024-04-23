# ImagePoint_Align

* 基于手动选择特征点的图像对齐

## 文件结构
```shell
├─demo             # 用于手动选择特征点图像对齐的算法示例
│  ├─ Align_insert          	  	# 直接图像对齐(居于选定的矩形)
│  │  ├─ Align_insert_v1.py
│  │  └─ Align_insert_v2.py
│  ├─ Align_warpAffline          # 基于仿射变换的对齐(三点)
│  │  ├─ Align_warpAffline_v1.py
│  │  └─ Align_warpAffline_v2.py
│  ├─ Align_Perspective          # 基于透视变换的对齐(四点)
│  │  └─ Align_Perspective_v1.py
│  ├─ Align_Homography          	# 基于单应性变换
│  │  └─ Align_insert_v1.py
│  ├─ image_files                # 用于对齐的图像示例
│  ├─ run_result                 # 不同算法的运行结果
│  ├─ points_extract_IR.py       # 红外图像提取特征点
│  ├─ points_extract_VIS.py      # 可见光图像提取特征点
│  ├─ points_IR.txt              # 红外图像提取特征点结果
│  └─ points_VIS.txt             # 可见光图像提取特征点结果
│
├─utils      	                # 调用的功能函数
│  ├─util_dataset.py            # 构建数据集
│  ├─util_device.py        	# 运行设备 
│  ├─util_fusion.py             # 模型推理
│  ├─util_loss.py            	# 结构误差损失函数
│  ├─util_train.py            	# 训练用相关函数
│  └─utils.py                   # 其他功能函数
|
├─ others                        # 一些其他的可运行代码作为参考
│
└─ README.md                     # README

```



## 使用说明

#### 不同的文件夹对应不同的运行结果

* **demo**文件夹下是对各种算法的实例
* **utils_ImagePointAlign**文件夹下是整理成了一个Package
* **others**文件夹下其他的相关可运行代码

### demo 文件夹

* 运行 **points_extract_VIS.py, points_extract_VIS.py** 文件，手动选择两个图像中的特征点。
  * 特征点的保存路径可以在运行文件中设置
  * 手动选择有三种模式：选点，修正和展示
* 运行不同 **Align_xxx** 文件夹下的对齐算法，结果保存在 **run_result** 中。
  * Align_insert 是直接图像对齐，基于特征点所在矩形区域
  * Align_warpAffline 基于仿射变换的对齐(三点)
  * Align_Perspective 基于透视变换的对齐(四点)
  * Align_Homography 基于单应性变换

