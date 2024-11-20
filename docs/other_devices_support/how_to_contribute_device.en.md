# 1. More Device Support

The prosperity of the PaddlePaddle ecosystem is inseparable from the contributions of developers and users. We warmly welcome you to provide more hardware compatibility for PaddleX and greatly appreciate your feedback.

Currently, PaddleX supports Intel/Apple M series CPU, NVIDIA GPUs, XPU, Ascend NPU, Hygon DCU, and MLU. If the hardware you wish to support is not within the current scope, you can contribute by following the methods below.

## 1.1 Integration Device into PaddlePaddle Backend

The PaddlePaddle deep learning framework provides multiple integration solutions, including operator development and mapping, subgraph and whole graph integration, deep learning compiler backend integration, and open neural network format conversion. Hardware vendors can flexibly choose based on their chip architecture design and software stack maturity. For specific details, please refer to [PaddlePaddle Hardware Integration Solutions](https://www.paddlepaddle.org.cn/documentation/docs/en/develop/dev_guides/custom_device_docs/index_en.html).

## 1.2 Support for Various Kits

Since PaddleX is based on the PaddlePaddle model library, after the hardware completes the integration into the PaddlePaddle backend, select the corresponding kit to submit code based on the models already supported by the hardware to ensure that the relevant kits are adapted to the corresponding hardware. Refer to the contribution guides for each kit:

1. [PaddleClas](https://github.com/PaddlePaddle/PaddleClas/tree/develop)

2. [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection/tree/develop)

3. [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg/tree/develop)

4. [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR/tree/develop)

5. [PaddleTS](https://github.com/PaddlePaddle/PaddleTS/tree/main)

# 2. Updating PaddleX

After completing the hardware integration into PaddlePaddle and the various kits, you need to update the hardware recognition-related code and documentation in PaddleX.

## 2.1 Inference Support

### 2.1.1 Version Support (Optional)

If the relevant hardware has specific requirements for the PaddlePaddle version, you can make judgments based on device information and version information during initialization. The relevant code is located in the `_check_paddle_version` function in [PaddleX Initialization](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/__init__.py).

### 2.1.2 Setting Environment Variables (Optional)

If special environment variables need to be set when using the relevant hardware, you can modify the device environment setup code. The relevant code is located in the `set_env_for_device` function in [PaddleX Environment Variable Settings](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/utils/device.py).

### 2.1.3 Creating a Predictor

PaddleX's inference capability is provided based on the Paddle Inference Predictor. When creating a Predictor, you need to select different hardware based on device information and create passes. The relevant code is located in the `_create` function in [PaddleX Predictor Creation](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/inference/components/paddle_predictor/predictor.py).

### 2.1.4 Updating the Hardware Support List

When creating a Predictor, it will judge whether the device is already supported. The relevant code is located in the `SUPPORT_DEVICE` constant in [PaddleX Predictor Option](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/inference/utils/pp_option.py).

### 2.1.5 Updating the Multi-Hardware User Guide

Please update the PaddleX multi-hardware user guide and add the newly supported hardware information to the documentation. Both Chinese and English versions need to be updated. The Chinese version is [PaddleX Multi-Hardware Usage Guide](https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/other_devices_support/multi_devices_use_guide.md), and the English version is [PaddleX Multi-Hardware Usage Guide](https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/other_devices_support/multi_devices_use_guide.en.md).

### 2.1.6 Updating the Installation Tutorial

Please provide hardware-related installation tutorials in both Chinese and English. The Chinese version can refer to [Ascend NPU PaddlePaddle Installation Tutorial](https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/other_devices_support/paddlepaddle_install_NPU.md), and the English version can refer to [Ascend NPU PaddlePaddle Installation Tutorial](https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/other_devices_support/paddlepaddle_install_NPU.en.md).

### 2.1.7 Updating the Model List

Please provide a list of models supported by the hardware in both Chinese and English. The Chinese version can refer to [PaddleX Model List (Huawei Ascend NPU)](https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/support_list/model_list_npu.md), and the English version can refer to [PaddleX Model List (Huawei Ascend NPU)](https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/support_list/model_list_npu.en.md).

## 2.2 Training Support

TODO

# 3. Submitting a PR

When you complete the hardware adaptation work, please submit a [Pull Request](https://github.com/PaddlePaddle/PaddleX/pulls) to PaddleX with relevant information. We will validate the model and merge the relevant code after confirmation.

The relevant PR needs to provide information on reproducing model accuracy, including at least the following:

* The software versions used to validate model accuracy, including but not limited to:

  * Paddle version

  * PaddleCustomDevice version (if any)

  * The branch of PaddleX or the corresponding kit

* The machine environment used to validate model accuracy, including but not limited to:

  * Chip model

  * Operating system version

  * Hardware driver version

  * Operator library version, etc.
