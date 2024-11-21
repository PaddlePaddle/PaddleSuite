# 1. 更多硬件支持

飞桨生态的繁荣离不开开发者和用户的贡献，我们非常欢迎您为 PaddleX 提供更多的硬件适配，也十分感谢您的反馈。

当前支持的类型为 Intel/苹果M系列 CPU、英伟达 GPU、昆仑芯 XPU、昇腾 NPU、海光 DCU 和寒武纪MLU，如果您要支持的硬件不在已经支持的范围内，可以参考如下方式进行贡献。

## 1.1 硬件接入飞桨后端

飞桨深度学习框架提供了多种硬件接入的方案，包括算子开发与映射、子图与整图接入、深度学习编译器后端接入以及开源神经网络格式转化四种硬件接入方案，供硬件厂商根据自身芯片架构设计与软件栈的建设成熟度进行灵活选择，具体细节请参考[飞桨硬件接入方案](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/custom_device_docs/index_cn.html)。

## 1.2 各个套件支持

由于PaddleX基于飞桨模型库实现。当硬件完成飞桨后端接入后，按照硬件已经支持的模型情况，选择相应的套件去提交代码，确保相关套件适配了对应硬件，参考各套件贡献指南：

1. [PaddleClas](https://github.com/PaddlePaddle/PaddleClas/tree/develop)

2. [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection/tree/develop)

3. [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg/tree/develop)

4. [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR/tree/develop)

5. [PaddleTS](https://github.com/PaddlePaddle/PaddleTS/tree/main)

# 2. 更新PaddleX

当完成硬件接入飞桨和各个套件后，需要更新PaddleX中硬件识别相关的代码和说明文档

## 2.1 推理能力支持

### 2.1.1 版本支持（可忽略）

如果相关硬件对于飞桨版本有特定要求，可以在初始化时根据设备信息和版本信息进行判断，相关代码位于 [PaddleX初始化](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/__init__.py)中的 `_check_paddle_version`

### 2.1.2 设置环境变量（可忽略）

如果相关硬件在使用时，需要设定特殊的环境变量，可以修改设备环境设置代码，相关代码位于 [PaddleX环境变量设置](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/utils/device.py)中的 `set_env_for_device`

### 2.1.3 创建Predictor

PaddleX的推理能力基于飞桨Paddle Inference Predictor提供，创建Predictor时需要根据设备信息选择不同的硬件并创建pass，相关代码位于[PaddleX Predictor创建](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/inference/components/paddle_predictor/predictor.py)的 `_create`

### 2.1.4 更新硬件支持列表

创建Predictor时会判断设备是否已支持，相关代码位于[PaddleX Predictor Option](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/inference/utils/pp_option.py)中的 `SUPPORT_DEVICE`

### 2.1.4 更新多硬件说明指南

请更新PaddleX多硬件说明指南，将新支持的硬件信息更新到文档中，需要同时更新中英文版本，中文版本 [PaddleX多硬件使用指南](https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/other_devices_support/multi_devices_use_guide.md) ，英文版本 [PaddleX Multi-Hardware Usage Guide](https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/other_devices_support/multi_devices_use_guide.en.md)

### 2.1.5 更新安装教程

请提供硬件相关的安装教程，需要提供中英文版本，中文版本参考 [昇腾 NPU 飞桨安装教程](https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/other_devices_support/paddlepaddle_install_NPU.md) ，英文版本参考 [Ascend NPU PaddlePaddle Installation Tutorial](https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/other_devices_support/paddlepaddle_install_NPU.en.md)

### 2.1.6 更新模型列表

请提供硬件支持的模型列表，需要提供中英文版本，中文版本参考 [PaddleX模型列表（昇腾 NPU）](https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/support_list/model_list_npu.md) ，英文版本参考 [PaddleX Model List (Huawei Ascend NPU)](https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/support_list/model_list_npu.en.md)

## 2.2 训练能力支持

TODO

# 3. 提交PR

当您完成特定硬件的适配工作后，请给PaddleX提交一个 [Pull Request](https://github.com/PaddlePaddle/PaddleX/pulls) 说明相关信息，我们将会对模型进行验证，确认无问题后将合入相关代码

相关PR需要提供复现模型精度的信息，至少包含以下内容：

* 验证模型精度所用到的软件版本，包括但不限于：

  * Paddle版本

  * PaddleCustomDevice版本（如果有）

  * PaddleX或者对应套件的分支

* 验证模型精度所用到的机器环境，包括但不限于：

  * 芯片型号

  * 系统版本

  * 硬件驱动版本

  * 算子库版本等