简体中文 | [English](pipeline_develop_guide_en.md)

# PaddleX模型产线使用概览

若您已经体验过PaddleX中的预训练模型产线效果，希望直接进行模型微调，可以跳转到[模型选择](#3模型选择可选)

完整的PaddleX模型产线开发流程如下图所示：

**<center>PaddleX 模型产线开发流程图</center>**

```mermaid
graph LR
    select_pipeline(选择产线) --> online_experience[快速体验]
    online_experience --> online_ok{效果满意？}
    online_ok --不满意--> select_model[选择模型]
    select_model --> model_finetune[模型微调]
    online_ok --满意--> development_integration(开发集成/部署)
    model_finetune --> pipeline_test[产线测试]
    pipeline_test --> test_ok{效果满意？}
    test_ok --不满意--> select_model
    test_ok --满意--> development_integration
```

PaddleX 所提供的预训练的模型产线均可以**快速体验效果**，如果产线效果可以达到您的要求，您可以直接将预训练的模型产线进行**开发集成/部署**，如果效果不及预期，可以使用私有数据对产线中的模型进行**微调**，直到达到满意的效果。

下面，让我们以登机牌识别的任务为例，介绍PaddleX模型产线工具的本地使用过程，在使用前，请确保您已经按照[PaddleX本地安装教程](../installation/installation.md)完成了PaddleX的安装。

## 1、选择产线

PaddleX中每条产线都可以解决特定任务场景的问题如目标检测、时序预测、语义分割等，您需要根据具体任务选择后续进行开发的产线。例如此处为登机牌识别任务，对应 PaddleX 的【通用 OCR 产线】。更多任务与产线的对应关系可以在 [PaddleX产线列表(CPU/GPU)](../support_list/pipelines_list.md)查询。

## 2、快速体验

PaddleX的每条产线都集成了众多预训练模型，您可以先体验PaddleX的预训练模型产线的效果，如果预训练模型产线的效果符合您的预期，即可直接进行[开发集成/部署](#6开发集成部署)，如果不符合，再根据后续步骤对产线的效果进行优化。

PaddleX提供了三种可以快速体验产线效果的方式，您可以根据需要选择合适的方式进行产线效果体验：

* 在线快速体验地址：[PaddleX产线列表(CPU/GPU)](../support_list/pipelines_list.md)
* 命令行快速体验：[PaddleX产线命令行使用说明](../pipeline_usage/instructions/pipeline_CLI_usage.md)
* Python脚本快速体验：[PaddleX产线Python脚本使用说明](../pipeline_usage/instructions/pipeline_python_API.md)
* 
以实现登机牌识别任务的通用OCR产线为例，一行命令即可快速体验产线效果：

```bash
paddlex --pipeline OCR --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png --device gpu:0
```
参数说明：

```bash
--pipeline：产线名称，此处为OCR产线
--input：待处理的输入图片的本地路径或URL
--device 使用的GPU序号（例如gpu:0表示使用第0块GPU，gpu:1,2表示使用第1、2块GPU），也可选择使用CPU（--device cpu）
```
执行后，将提示选择 OCR 产线配置文件保存路径，默认保存至*当前目录*，也可 *自定义路径*。[@郜廷权](https://ku.baidu-int.com?t=mention&mt=contact&id=0e9e1070-7ca6-11ef-928f-85a316a9b6e7)

此外，也可在执行命令时加入 `-y` 参数，则可跳过路径选择，直接将产线配置文件保存至当前目录。

获取产线配置文件后，可将 `--pipeline` 替换为配置文件保存路径，即可使配置文件生效。例如，若配置文件保存路径为 `./ocr.yaml`，只需执行：

```bash
paddlex --pipeline ./ocr.yaml --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png
```
其中，`--model`、`--device` 等参数无需指定，将使用配置文件中的参数。若依然指定了参数，将以指定的参数为准。

运行后，得到的结果为：

```bash
The prediction result is:
['登机口于起飞前10分钟关闭']
The prediction result is:
['GATES CLOSE 1O MINUTESBEFORE DEPARTURE TIME']
The prediction result is:
['ETKT7813699238489/1']
......
```
可视化结果如下：

![](https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/boardingpass.png)
## 3、模型选择（可选）

由于一个产线中可能包含一个或多个模型，在进行模型微调时，您需要根据测试的情况确定微调其中的哪个模型。以此处登机牌识别任务的OCR产线为例，该产线包含文本检测模型（如 `PP-OCRv4_mobile_det`）和文本识别模型（如 `PP-OCRv4_mobile_rec`），如发现文字的定位不准，则需要微调文本检测模型，如果发现文字的识别不准，则需要微调文本识别模型。如果您不清楚产线中包含哪些模型，可以查阅 [PaddleX产线列表(CPU/GPU)](../support_list/pipelines_list.md)

## 4、模型微调（可选）

在确定好需要微调的模型后，您需要用私有数据集对模型进行训练，PaddleX提供了单模型开发工具，一行命令即可完成模型的训练：

```bash
python main.py -c paddlex/configs/text_recognition/PP-OCRv4_mobile_rec.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=your/dataset_dir
```
此外，对于模型微调中私有数据集的准备、单模型的推理等内容，PaddleX也提供了详细的教程，详细请参考[PaddleX单模型开发工具使用教程](../module_usage/module_develop_guide.md)

## 5、产线测试（可选）

当您使用私有数据集完成微调训练后，可获得本地模型权重文件。

若您需要使用微调后的模型权重，只需对产线配置文件做修改，将微调后模型权重的本地路径替换至产线配置文件中的对应位置即可：

```bash
......
Pipeline:
  det_model: PP-OCRv4_server_det  #可修改为微调后文本检测模型的本地路径
  det_device: "gpu"
  rec_model: PP-OCRv4_server_rec  #可修改为微调后文本识别模型的本地路径
  rec_batch_size: 1
  rec_device: "gpu"
......
```
随后， 参考[快速体验](#2快速体验)中的命令行方式或Python脚本方式，加载修改后的产线配置文件即可。

如果效果满意，即可将微调后的产线进行[开发集成/部署](#6开发集成部署)，如果不满意，即可回到[模型选择](#3模型选择可选)尝试继续微调其他任务模块的模型，直到达到满意的效果。

## 6、开发集成/部署

PaddleX提供了简洁的Python API，用几行代码即可将模型产线集成到您的项目中。此处用于集成登机牌识别的OCR产线示例代码如下：

```bash
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="OCR")
output = pipeline.predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_001.png")
for res in output:
    res.print(json_format=False)
    res.save_to_img("./output/")
    res.save_to_json("./output/res.json")
```
更详细的模型产线Python集成方法请参考[PaddleX产线Python脚本使用说明](../pipeline_usage/instructions/pipeline_python_API.md)

同时，PaddleX提供了三种部署方式及详细的部署教程：

🚀 **高性能部署**：在实际生产环境中，许多应用对部署策略的性能指标（尤其是响应速度）有着较严苛的标准，以确保系统的高效运行与用户体验的流畅性。为此，PaddleX 提供高性能推理插件，旨在对模型推理及前后处理进行深度性能优化，实现端到端流程的显著提速，详细的高性能部署流程请参考[PaddleX高性能部署指南](../pipeline_deploy/high_performance_deploy.md)。

☁️ **服务化部署**：服务化部署是实际生产环境中常见的一种部署形式。通过将推理功能封装为服务，客户端可以通过网络请求来访问这些服务，以获取推理结果。PaddleX 支持用户以低成本实现产线的服务化部署，详细的服务化部署流程请参考[PaddleX服务化部署指南](../pipeline_deploy/service_deploy.md)。

📱 **端侧部署**：端侧部署是一种将计算和数据处理功能放在用户设备本身上的方式，设备可以直接处理数据，而不需要依赖远程的服务器。PaddleX 支持将模型部署在 Android 等端侧设备上，详细的端侧部署流程请参考[PaddleX端侧部署指南](../pipeline_deploy/lite_deploy.md)。
您可以根据需要选择合适的方式部署模型产线，进而进行后续的 AI 应用集成。



> ❗ 温馨提醒：PaddleX为每个产线都提供了详细的使用说明，您可以根据需要进行选择，所有产线对应的使用说明如下：

| 产线名称           | 详细说明                                                                                                      |
|--------------------|----------------------------------------------------------------------------------------------------------------|
| 文档场景信息抽取v3   | [文档场景信息抽取v3产线Python脚本使用说明](/docs_new/pipeline_usage/tutorials/information_extration_pipelines/document_scene_information_extraction.md) |
| 通用图像分类       | [通用图像分类产线Python脚本使用说明](/docs_new/pipeline_usage/tutorials/cv_pipelines/image_classification.md) |
| 通用目标检测       | [通用目标检测产线Python脚本使用说明](/docs_new/pipeline_usage/tutorials/cv_pipelines/image_classification.md) |
| 通用实例分割       | [通用实例分割产线Python脚本使用说明](/docs_new/pipeline_usage/tutorials/cv_pipelines/instance_segmentation.md) |
| 通用语义分割       | [通用语义分割产线Python脚本使用说明](/docs_new/pipeline_usage/tutorials/cv_pipelines/semantic_segmentation.md) |
| 通用图像多标签分类 | [通用图像多标签分类产线Python脚本使用说明](/docs_new/pipeline_usage/tutorials/cv_pipelines/image_multi_label_lassification.md) |
| 小目标检测         |  [小目标检测产线Python脚本使用说明](/docs_new/pipeline_usage/tutorials/cv_pipelines/small_object_detection.md) |
| 图像异常检测       | [图像异常检测产线Python脚本使用说明](/docs_new/pipeline_usage/tutorials/cv_pipelines/image_anomaly_detection.md) |
| 通用OCR            | [通用OCR产线Python脚本使用说明](/docs_new/pipeline_usage/tutorials/ocr_pipelies/OCR.md) |
| 通用表格识别       | [通用表格识别产线Python脚本使用说明](/docs_new/pipeline_usage/tutorials/ocr_pipelies/table_recognition.md) |
| 通用时序预测       | [通用时序预测产线Python脚本使用说明](/docs_new/pipeline_usage/tutorials/time_series_pipelines/time_series_forecasting.md) |
| 通用时序异常检测   | [通用时序异常检测产线Python脚本使用说明](/docs_new/pipeline_usage/tutorials/time_series_pipelines/time_series_anomaly_detection.md) |
| 通用时序分类       | [通用时序分类产线Python脚本使用说明](/docs_new/pipeline_usage/tutorials/time_series_pipelines/time_series_classification.md) |