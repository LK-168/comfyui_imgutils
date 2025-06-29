# comfyui_imgutils

这个是 ComfyUI 的图像处理工具集，提供了多种图像处理和检测功能

部分功能基于imgutils库
https://github.com/deepghs/imgutils
模型文件全部由该库提供和管理下载，会自动下载到 
`HF_HOME` 环境变量指定的目录下
因此需要在 ComfyUI 的启动项中设置该环境变量


Segment-Anything 模型需要手动下载到
`ComfyUI\models\sams`目录下

由于我没研究明白的原因，该路径不受`extra_model_paths.yaml`的影响（如果你有解决方案欢迎 PR）

如果你使用了 impact-pack 的 sam 加载器，恭喜你不需要重新下载模型了，因为这部分我就是抄的人家的代码，你甚至可以用人家的加载器来加载模型

============================= LK ==================================


节点介绍：

#### 检测节点
- Imgutils Generic Detector
    - 支持多种 `imgutils` 提供的多种基于anime的检测模型
        - `detection_type`：检测类型     
        - `conf_threshold`：置信度阈值
        - `iou_threshold`：IOU阈值, 用于非极大值抑制，总之就是用于合并重叠的检测框的
        - `draw_boxes`: 是否在输出的图片上绘制检测框
        - `level`: 模型的类型 `n` （nano） 和 `s` （standard）两种
        - `version`: 模型的版本，不填是用默认的，我之后会改的好一点

- Mask to BBox 、 BBox to Mask
    - 用于 `Mask` 和 `BBox` 之间的转换
- BBoxFilter
    - 用于过滤 `BBox`，可以根据置信度、面积和 标签进行过滤
        - `labels`的值依照的是 `Imgutils Generic Detector` 输出的图片中bbox上标注的标签，可以输入多个，用逗号分隔

#### segment-anything节点
基本上抄 Impact-Pack 的

#### segment 节点
- Imgutils Auto Segmenter
    - 仅能对图片进行 **前景和背景** 的分割

- Imgutils BBox Segmenter
    - 在 `Imgutils Generic Detector` 的基础上，使用检测到的 `BBox` 进行分割

##### 打码节点
- Censor with Mask
    - 利用提供的 `Mask` 对图片进行打码处理
        - `censor_type`：打码类型
            - `blur`：模糊
            - `pixelate`：像素化
            - `color`：纯色遮挡
 
#### Mask处理节点
收录一些常用的 Mask 处理节点
- Mask Morphology:
    - 提供了常用的形态学操作：膨胀、腐蚀、开运算和闭运算
    - 
- Mask Edge Operations:
    - 提供了常用的边缘处理操作：：扩展、收缩、扩展收缩和收缩扩展

- Mask Attributes:
    - 提供了常用的属性调整操作：二值化、阈值处理、平滑处理、对比度和亮度调整、反转和二值化
- Mask Combine:
    - 提供了多mask合并操作：加法、减法、乘法、除法、最大值、最小值和异或操作
- Mask Info:
    - 显示mask的统计信息，如形状、覆盖率、范围和均值
- MaskHelperLK:
    - 如果你忘记了以上`Mask`处理节点的功能，可以使用这个节点查看，因为我知道这个东西只看节点名字很难知道效果


懒了，今天先写到这，传播民主与自由去了。

