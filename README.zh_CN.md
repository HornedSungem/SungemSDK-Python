<div align="center">
    <a href="http://hornedsungem.org/">
        <img src="https://raw.githubusercontent.com/HornedSungem/SungemSDK/master/logo/logo.png" style="max-height:220px"/>
    </a>
    <p style="font-size:xx-large">Horned Sungem</p>
</div>

![python](https://img.shields.io/badge/python-3.x-blue.svg)

# SungemSDK-Python

[SungemSDK] | [文档中心][Documentation]

## 安装说明

通过如下命令获取项目：

```bash
git clone https://github.com/HornedSungem/SungemSDK-Python.git
```

进入 `installer` 文件夹，找到和您的操作系统对应的安装脚本，执行安装。

```bash
# 以 Ubuntu 为例
cd installer/Linux/Ubuntu
./install.sh
```

## 开始使用

本项目为 `Python` 开发者提供了API和一些示例程序。

### 示例程序

在运行示例程序之前，需要从 [SungemSDK-GraphModels][] 下载需要的文件，并拷贝至本项目对应的文件夹下。

```
SungemSDK-GraphModels/graphs -> examples/graphs
SungemSDK-GraphModels/misc   -> examples/misc
```

之后就可以运行示例程序了。

```bash
# 以 FaceDetector 为例
cd examples/apps/FaceDetector
python3 FaceDetector.py
```

## 联系我们

如果你需要帮助，欢迎在 [GitHub Issues][] 给我们提问，或者通过邮件(support@hornedsungem.org)与我们进行沟通。

如有任何建议、模型需求或希望加入我们也欢迎和我们联系。


[GitHub Issues]: https://github.com/HornedSungem/SungemSDK-Python/issues
[SungemSDK]: https://github.com/HornedSungem/SungemSDK
[Documentation]: https://hornedsungem.github.io/Docs
[SungemSDK-GraphModels]: https://github.com/HornedSungem/SungemSDK-GraphModels