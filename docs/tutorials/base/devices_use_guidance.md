# 多硬件使用指南

本文档主要针对昇腾 NPU、寒武纪 MLU、昆仑 XPU 硬件平台，介绍PaddleX使用指南。

## 1、硬件环境准备

### 1.1 飞桨安装

根据所属硬件平台拉取镜像、安装飞桨，请参考[多硬件飞桨安装](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta/docs/tutorials/INSTALL_OTHER_DEVICES.md)

### 1.2 PaddleX安装

#### 1.2.1 获取源码

##### 【推荐】从 GitHub 下载

使用下述命令从 GitHub 获取 PaddleX 最新源码。

```bash
git clone https://github.com/PaddlePaddle/PaddleX.git
```

##### 从 Gitee 下载

如果访问 GitHub 网速较慢，可以从 Gitee 下载，命令如下：

```bash
git clone https://gitee.com/paddlepaddle/PaddleX.git
```

#### 1.2.2 安装配置及依赖

参考下述命令，按提示操作，完成 PaddleX 依赖的安装。

```bash
cd PaddleX

# 注意，如果是arm64架构的xpu机器，需要先手动安装tool_helpers，x86架构机器不需要
pip install https://paddle-model-ecology.bj.bcebos.com/paddlex/whl/paddlenlp-device/tool_helpers-0.1.1-cp310-cp310-linux_aarch64.whl

# 安装 PaddleX whl
# -e：以可编辑模式安装，当前项目的代码更改，都会直接作用到已经安装的 PaddleX Wheel
pip install -e .

# 安装 PaddleX 相关依赖
paddlex --install

# 完成安装后会有如下提示：
# All packages are installed.
```

## 2、PaddleX使用

paddleX 模型训练/评估/推理的详细使用方法，参考文档：[模型训练/评估/推理](./docs/tutorials/base/README.md)
在具体设备上，根据所属硬件平台，添加配置设备的参数，即可在对应硬件上使用上述工具。
在xpu机器上使用0、1、2、3卡进行 PaddleX 模型的训练，可参考如下命令：

```bash
python main.py -c paddlex/configs/image_classification/PP-LCNet_x1_0.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/cls_flowers_examples \
    -o Global.device=xpu:0,1,2,3
```

评估、推理命令同理。
使用产线推理时，可参考如下命令：

```bash
paddlex --pipeline image_classification --model PP-LCNet_x1_0 --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg --device xpu:0
```
