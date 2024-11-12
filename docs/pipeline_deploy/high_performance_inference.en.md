---
comments: true
---

# PaddleX High-Performance Inference Guide

In real-world production environments, many applications have stringent standards for deployment strategy performance metrics, particularly response speed, to ensure efficient system operation and smooth user experience. To this end, PaddleX provides high-performance inference plugins designed to deeply optimize model inference and pre/post-processing, achieving significant speedups in the end-to-end process. This document will first introduce the installation and usage of the high-performance inference plugins, followed by a list of pipelines and models currently supporting the use of these plugins.

## 1. Installation and Usage of High-Performance Inference Plugins

Before using the high-performance inference plugins, ensure you have completed the installation of PaddleX according to the [PaddleX Local Installation Tutorial](../installation/installation.en.md), and have successfully run the quick inference of the pipeline using either the PaddleX pipeline command line instructions or the Python script instructions.

### 1.1 Installing High-Performance Inference Plugins

Find the corresponding installation command based on your processor architecture, operating system, device type, and Python version in the table below and execute it in your deployment environment. Please replace `{paddlex version number}` with the actual paddlex version number, such as the current latest stable version `3.0.0b2`. If you need to use the version corresponding to the development branch, replace `{paddlex version number}` with `0.0.0.dev0`.

<table>
  <tr>
    <th>Processor Architecture</th>
    <th>Operating System</th>
    <th>Device Type</th>
    <th>Python Version</th>
    <th>Installation Command</th>
  </tr>
  <tr>
    <td rowspan="7">x86-64</td>
    <td rowspan="7">Linux</td>
    <td rowspan="4">CPU</td>
  </tr>
  <tr>
    <td>3.8</td>
    <td>curl -s https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hpi/install_script/{paddlex version number}/install_paddlex_hpi.py | python3.8 - --arch x86_64 --os linux --device cpu --py 38</td>
  </tr>
  <tr>
    <td>3.9</td>
    <td>curl -s https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hpi/install_script/{paddlex version number}/install_paddlex_hpi.py | python3.9 - --arch x86_64 --os linux --device cpu --py 39</td>
  </tr>
  <tr>
    <td>3.10</td>
    <td>curl -s https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hpi/install_script/{paddlex version number}/install_paddlex_hpi.py | python3.10 - --arch x86_64 --os linux --device cpu --py 310</td>
  </tr>
  <tr>
    <td rowspan="3">GPU&nbsp;(CUDA&nbsp;11.8&nbsp;+&nbsp;cuDNN&nbsp;8.6)</td>
    <td>3.8</td>
    <td>curl -s https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hpi/install_script/{paddlex version number}/install_paddlex_hpi.py | python3.8 - --arch x86_64 --os linux --device gpu_cuda118_cudnn86 --py 38</td>
  </tr>
  <tr>
    <td>3.9</td>
    <td>curl -s https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hpi/install_script/{paddlex version number}/install_paddlex_hpi.py | python3.9 - --arch x86_64 --os linux --device gpu_cuda118_cudnn86 --py 39</td>
  </tr>
  <tr>
    <td>3.10</td>
    <td>curl -s https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hpi/install_script/{paddlex version number}/install_paddlex_hpi.py | python3.10 - --arch x86_64 --os linux --device gpu_cuda118_cudnn86 --py 310</td>
  </tr>
</table>

* For Linux systems, execute the installation instructions using Bash.
* When using NVIDIA GPUs, please use the installation instructions corresponding to the CUDA and cuDNN versions that match your environment. Otherwise, you will not be able to use the high-performance inference plugin properly.
* When the device type is CPU, the installed high-performance inference plugin only supports inference using the CPU; for other device types, the installed high-performance inference plugin supports inference using the CPU or other devices.

### 1.2 Obtaining Serial Numbers and Activation

On the [Baidu AIStudio Community - AI Learning and Training Platform](https://aistudio.baidu.com/paddlex/commercialization) page, under the "Open-source Pipeline Deployment Serial Number Inquiry and Acquisition" section, select "Acquire Now" as shown in the following image:

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipeline_deploy/image-1.png">

Select the pipeline you wish to deploy and click "Acquire". Afterwards, you can find the acquired serial number in the "Open-source Pipeline Deployment SDK Serial Number Management" section at the bottom of the page:

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipeline_deploy/image-2.png">

After using the serial number to complete activation, you can utilize high-performance inference plugins. PaddleX provides both online and offline activation methods (both only support Linux systems):

* Online Activation: When using the inference API or CLI, specify the serial number and enable online activation to automatically complete the process.
* Offline Activation: Follow the instructions in the serial number management interface (click "Offline Activation" under "Operations") to obtain the device fingerprint of your machine. Bind the serial number with the device fingerprint to obtain a certificate and complete the activation. For this activation method, you need to manually store the certificate in the `${HOME}/.baidu/paddlex/licenses` directory on the machine (create the directory if it does not exist) and specify the serial number when using the inference API or CLI.

Please note: Each serial number can only be bound to a unique device fingerprint and can only be bound once. This means that if users deploy models on different machines, they must prepare separate serial numbers for each machine.

### 1.3 Enabling High-Performance Inference Plugins

For Linux systems, if using the high-performance inference plugin in a Docker container, please mount the host machine's `/dev/disk/by-uuid` and `${HOME}/.baidu/paddlex/licenses` directories to the container.

For PaddleX CLI, specify `--use_hpip` and set the serial number to enable the high-performance inference plugin. If you wish to activate the license online, specify `--update_license` when using the serial number for the first time. Taking the general image classification pipeline as an example:

```bash
paddlex \
    --pipeline image_classification \
    --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg \
    --device gpu:0 \
    --use_hpip \
    --serial_number {serial_number}

# If you wish to perform online activation
paddlex \
    --pipeline image_classification \
    --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg \
    --device gpu:0 \
    --use_hpip \
    --serial_number {serial_number} \
    --update_license
```

For PaddleX Python API, enabling the high-performance inference plugin is similar. Still taking the general image classification pipeline as an example:

```python
from paddlex import create_pipeline

pipeline = create_pipeline(
    pipeline="image_classification",
    use_hpip=True,
    hpi_params={"serial_number": "{serial_number}"},
)

output = pipeline.predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg")
```

The inference results obtained with the high-performance inference plugin enabled are consistent with those without the plugin enabled. For some models, enabling the high-performance inference plugin for the first time may take a longer time to complete the construction of the inference engine. PaddleX will cache the relevant information in the model directory after the first construction of the inference engine and reuse the cached content in subsequent runs to improve initialization speed.

### 1.4 Modifying High-Performance Inference Configurations

PaddleX combines model information and runtime environment information to provide default high-performance inference configurations for each model. These default configurations are carefully prepared to be applicable in several common scenarios and achieve relatively optimal performance. Therefore, users typically may not need to be concerned with the specific details of these configurations. However, due to the diversity of actual deployment environments and requirements, the default configuration may not yield ideal performance in certain scenarios and could even result in inference failures. In cases where the default configuration does not meet the requirements, users can manually adjust the configuration by modifying the Hpi field in the inference.yml file within the model directory (if this field does not exist, it needs to be added). The following are two common situations:

- Switching inference backends:

    When the default inference backend is not available, the inference backend needs to be switched manually. Users should modify the `selected_backends` field (if it does not exist, it needs to be added).

    ```yaml
    Hpi:
      ...
      selected_backends:
        cpu: paddle_infer
        gpu: onnx_runtime
      ...
    ```

    Each entry should follow the format `{device type}: {inference backend name}`.

    The currently available inference backends are:

    * `paddle_infer`: The Paddle Inference engine. Supports CPU and GPU. Compared to the PaddleX quick inference, TensorRT subgraphs can be integrated to enhance inference performance on GPUs.
    * `openvino`: [OpenVINO](https://github.com/openvinotoolkit/openvino), a deep learning inference tool provided by Intel, optimized for model inference performance on various Intel hardware. Supports CPU only. The high-performance inference plugin automatically converts the model to the ONNX format and uses this engine for inference.
    * `onnx_runtime`: [ONNX Runtime](https://onnxruntime.ai/), a cross-platform, high-performance inference engine. Supports CPU and GPU. The high-performance inference plugin automatically converts the model to the ONNX format and uses this engine for inference.
    * `tensorrt`: [TensorRT](https://developer.nvidia.com/tensorrt), a high-performance deep learning inference library provided by NVIDIA, optimized for NVIDIA GPUs to improve speed. Supports GPU only. The high-performance inference plugin automatically converts the model to the ONNX format and uses this engine for inference.

- Modifying dynamic shape configurations for Paddle Inference or TensorRT:

    Dynamic shape is the ability of TensorRT to defer specifying parts or all of a tensor’s dimensions until runtime. If the default dynamic shape configuration does not meet requirements (e.g., the model may require input shapes beyond the default range), users need to modify the `trt_dynamic_shapes` or `dynamic_shapes` field in the inference backend configuration:

    ```yaml
    Hpi:
      ...
      backend_configs:
        # Configuration for the Paddle Inference backend
        paddle_infer:
          ...
          trt_dynamic_shapes:
            x:
              - [1, 3, 300, 300]
              - [4, 3, 300, 300]
              - [32, 3, 1200, 1200]
          ...
        # Configuration for the TensorRT backend
        tensorrt:
          ...
          dynamic_shapes:
            x:
              - [1, 3, 300, 300]
              - [4, 3, 300, 300]
              - [32, 3, 1200, 1200]
          ...
    ```

    In `trt_dynamic_shapes` or `dynamic_shapes`, each input tensor requires a specified dynamic shape in the format: `{input tensor name}: [{minimum shape}, [{optimal shape}], [{maximum shape}]]`. For details on minimum, optimal, and maximum shapes and further information, please refer to the official TensorRT documentation.

    After completing the modifications, please delete the cache files in the model directory (`shape_range_info.pbtxt` and files starting with `trt_serialized`).

## 2. Pipelines and Models Supporting High-Performance Inference Plugins

<table>
  <tr>
    <th>Pipeline</th>
    <th>Pipeline Module</th>
    <th>Specific Models</th>
  </tr>
  <tr>
    <td>General Image Classification</td>
    <td>Image Classification</td>
    <td>ResNet18<br/>ResNet34<details>
    <summary><b>more</b></summary>ResNet50<br/>ResNet101<br/>ResNet152<br/>ResNet18_vd<br/>ResNet34_vd<br/>ResNet50_vd<br/>ResNet101_vd<br/>ResNet152_vd<br/>ResNet200_vd<br/>PP-LCNet_x0_25<br/>PP-LCNet_x0_35<br/>PP-LCNet_x0_5<br/>PP-LCNet_x0_75<br/>PP-LCNet_x1_0<br/>PP-LCNet_x1_5<br/>PP-LCNet_x2_0<br/>PP-LCNet_x2_5<br/>PP-LCNetV2_small<br/>PP-LCNetV2_base<br/>PP-LCNetV2_large<br/>MobileNetV3_large_x0_35<br/>MobileNetV3_large_x0_5<br/>MobileNetV3_large_x0_75<br/>MobileNetV3_large_x1_0<br/>MobileNetV3_large_x1_25<br/>MobileNetV3_small_x0_35<br/>MobileNetV3_small_x0_5<br/>MobileNetV3_small_x0_75<br/>MobileNetV3_small_x1_0<br/>MobileNetV3_small_x1_25<br/>ConvNeXt_tiny<br/>ConvNeXt_small<br/>ConvNeXt_base_224<br/>ConvNeXt_base_384<br/>ConvNeXt_large_224<br/>ConvNeXt_large_384<br/>MobileNetV1_x0_25<br/>MobileNetV1_x0_5<br/>MobileNetV1_x0_75<br/>MobileNetV1_x1_0<br/>MobileNetV2_x0_25<br/>MobileNetV2_x0_5<br/>MobileNetV2_x1_0<br/>MobileNetV2_x1_5<br/>MobileNetV2_x2_0<br/>SwinTransformer_tiny_patch4_window7_224<br/>SwinTransformer_small_patch4_window7_224<br/>SwinTransformer_base_patch4_window7_224<br/>SwinTransformer_base_patch4_window12_384<br/>SwinTransformer_large_patch4_window7_224<br/>SwinTransformer_large_patch4_window12_384<br/>PP-HGNet_small<br/>PP-HGNet_tiny<br/>PP-HGNet_base<br/>PP-HGNetV2-B0<br/>PP-HGNetV2-B1<br/>PP-HGNetV2-B2<br/>PP-HGNetV2-B3<br/>PP-HGNetV2-B4<br/>PP-HGNetV2-B5<br/>PP-HGNetV2-B6<br/>CLIP_vit_base_patch16_224<br/>CLIP_vit_large_patch14_224</details></td>
  </tr>

  <tr>
    <td>General Object Detection</td>
    <td>Object Detection</td>
    <td>PP-YOLOE_plus-S<br/>PP-YOLOE_plus-M<details>
        <summary><b>more</b></summary>PP-YOLOE_plus-L<br/>PP-YOLOE_plus-X<br/>YOLOX-N<br/>YOLOX-T<br/>YOLOX-S<br/>YOLOX-M<br/>YOLOX-L<br/>YOLOX-X<br/>YOLOv3-DarkNet53<br/>YOLOv3-ResNet50_vd_DCN<br/>YOLOv3-MobileNetV3<br/>RT-DETR-R18<br/>RT-DETR-R50<br/>RT-DETR-L<br/>RT-DETR-H<br/>RT-DETR-X<br/>PicoDet-S<br/>PicoDet-L</details></td>
  </tr>

  <tr>
    <td>General Semantic Segmentation</td>
    <td>Semantic Segmentation</td>
    <td>Deeplabv3-R50<br/>Deeplabv3-R101<details>
    <summary><b>more</b></summary>Deeplabv3_Plus-R50<br/>Deeplabv3_Plus-R101<br/>PP-LiteSeg-T<br/>OCRNet_HRNet-W48<br/>OCRNet_HRNet-W18<br/>SeaFormer_tiny<br/>SeaFormer_small<br/>SeaFormer_base<br/>SeaFormer_large<br/>SegFormer-B0<br/>SegFormer-B1<br/>SegFormer-B2<br/>SegFormer-B3<br/>SegFormer-B4<br/>SegFormer-B5</details></td>
  </tr>

  <tr>
    <td>General Instance Segmentation</td>
    <td>Instance Segmentation</td>
    <td>Mask-RT-DETR-L<br/>Mask-RT-DETR-H</td>
  </tr>

  <tr>
    <td rowspan="3">Seal Text Recognition</td>
    <td>Layout Analysis</td>
    <td>PicoDet-S_layout_3cls<br/>PicoDet-S_layout_17cls<details>
    <summary><b>more</b></summary>PicoDet-L_layout_3cls<br/>PicoDet-L_layout_17cls<br/>RT-DETR-H_layout_3cls<br/>RT-DETR-H_layout_17cls</details></td>
  </tr>

  <tr>
    <td>Seal Text Detection</td>
    <td>PP-OCRv4_server_seal_det<br/>PP-OCRv4_mobile_seal_det</td>
  </tr>

  <tr>
    <td>Text Recognition</td>
    <td>PP-OCRv4_mobile_rec<br/>PP-OCRv4_server_rec</td>
  </tr>

  <tr>
    <td rowspan="2">General OCR</td>
    <td>Text Detection</td>
    <td>PP-OCRv4_server_det<br/>PP-OCRv4_mobile_det</td>
  </tr>

  <tr>
    <td>Text Recognition</td>
    <td>PP-OCRv4_server_rec<br/>PP-OCRv4_mobile_rec<br/>ch_RepSVTR_rec<br/>ch_SVTRv2_rec</td>
  </tr>

  <tr>
    <td rowspan="5">General Table Recognition</td>
    <td>Layout Detection</td>
    <td>PicoDet_layout_1x</td>
  </tr>

  <tr>
    <td rowspan="2">Table Recognition</td>
    <td>SLANet</td>
  </tr>

  <tr>
    <td>SLANet_plus</td>
  </tr>

  <tr>
    <td>Text Detection</td>
    <td>PP-OCRv4_server_det<br/>PP-OCRv4_mobile_det</td>
  </tr>

  <tr>
    <td>Text Recognition</td>
    <td>PP-OCRv4_server_rec<br/>PP-OCRv4_mobile_rec<br/>ch_RepSVTR_rec<br/>ch_SVTRv2_rec</td>
  </tr>

  <tr>
    <td rowspan="15">Document Scene Information Extraction v3</td>
    <td rowspan="2">Table Recognition</td>
    <td>SLANet</td>
  </tr>

  <tr>
    <td>SLANet_plus</td>
  </tr>

  <tr>
    <td>Layout Detection</td>
    <td>PicoDet_layout_1x</td>
  </tr>

  <tr>
    <td rowspan="2">Text Detection</td>
    <td>PP-OCRv4_server_det</td>
  </tr>

  <tr>
    <td>PP-OCRv4_mobile_det</td>
  </tr>

  <tr>
    <td rowspan="4">Text Recognition</td>
    <td>PP-OCRv4_server_rec</td>
  </tr>

  <tr>
    <td>PP-OCRv4_mobile_rec</td>
  </tr>

  <tr>
    <td>ch_RepSVTR_rec</td>
  </tr>

  <tr>
    <td>ch_SVTRv2_rec</td>
  </tr>

  <tr>
    <td rowspan="2">Seal Text Detection</td>
    <td>PP-OCRv4_server_seal_det</td>
  </tr>

  <tr>
    <td>PP-OCRv4_mobile_seal_det</td>
  </tr>

  <tr>
    <td>Text Image Rectification</td>
    <td>UVDoc</td>
  </tr>

  <tr>
    <td>Document Image Orientation Classification</td>
    <td>PP-LCNet_x1_0_doc_ori</td>
  </tr>

</table>
