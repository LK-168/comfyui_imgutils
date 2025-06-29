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

