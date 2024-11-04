简体中文 | [English](vehicle_attribute_en.md)

# 车辆属性识别产线使用教程

## 1. 车辆属性识别产线介绍
Stay tuned

## 2. 快速开始
PaddleX 所提供的预训练的模型产线均可以快速体验效果，你可以在线体验车辆属性识别产线的效果，也可以在本地使用命令行或 Python 体验车辆属性识别产线的效果。

### 2.1 在线体验
Stay tuned

### 2.2 本地体验
在本地使用车辆属性识别产线前，请确保您已经按照[PaddleX本地安装教程](../../../installation/installation.md)完成了PaddleX的wheel包安装。

#### 2.2.1 命令行方式体验
一行命令即可快速体验车辆属性识别产线效果，使用 [测试文件](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/vehicle_attribute_002.jpg)，并将 `--input` 替换为本地路径，进行预测

```bash
paddlex --pipeline vehicle_attribute --input vehicle_attribute_002.jpg --device gpu:0
```
参数说明：

```
--pipeline：产线名称，此处为车辆属性识别产线
--input：待处理的输入图片的本地路径或URL
--device 使用的GPU序号（例如gpu:0表示使用第0块GPU，gpu:1,2表示使用第1、2块GPU），也可选择使用CPU（--device cpu）
```

在执行上述 Python 脚本时，加载的是默认的车辆属性识别产线配置文件，若您需要自定义配置文件，可执行如下命令获取：

<details>
   <summary> 👉点击展开</summary>

```
paddlex --get_pipeline_config vehicle_attribute
```
执行后，车辆属性识别产线配置文件将被保存在当前路径。若您希望自定义保存位置，可执行如下命令（假设自定义保存位置为 `./my_path` ）：

```
paddlex --get_pipeline_config vehicle_attribute --save_path ./my_path
```

获取产线配置文件后，可将 `--pipeline` 替换为配置文件保存路径，即可使配置文件生效。例如，若配置文件保存路径为 `./vehicle_attribute.yaml`，只需执行：

```bash
paddlex --pipeline ./vehicle_attribute.yaml --input vehicle_attribute_002.jpg --device gpu:0
```
其中，`--model`、`--device` 等参数无需指定，将使用配置文件中的参数。若依然指定了参数，将以指定的参数为准。

</details>

运行后，得到的结果为：

```
{'input_path': 'vehicle_attribute_002.jpg', 'boxes': [{'labels': ['Trousers(长裤)', 'Age18-60(年龄在18-60岁之间)', 'LongCoat(长外套)', 'Side(侧面)'], 'cls_scores': array([0.99965, 0.99963, 0.98866, 0.9624 ]), 'det_score': 0.9795265793800354, 'coordinate': [87.24845, 322.57797, 546.27014, 1039.9806]}, {'labels': ['Trousers(长裤)', 'LongCoat(长外套)', 'Front(面朝前)', 'Age18-60(年龄在18-60岁之间)'], 'cls_scores': array([0.99996, 0.99872, 0.93379, 0.71614]), 'det_score': 0.9671529531478882, 'coordinate': [737.9159, 306.28375, 1150.6005, 1034.2983]}, {'labels': ['Trousers(长裤)', 'LongCoat(长外套)', 'Age18-60(年龄在18-60岁之间)', 'Side(侧面)'], 'cls_scores': array([0.99996, 0.99514, 0.98726, 0.96224]), 'det_score': 0.9645677208900452, 'coordinate': [399.46594, 281.90945, 869.5361, 1038.995]}]}
```

#### 2.2.2 Python脚本方式集成
几行代码即可完成产线的快速推理，以车辆属性识别产线为例：

```
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="vehicle_attribute")

output = pipeline.predict("vehicle_attribute_002.jpg")
for res in output:
    res.print() ## 打印预测的结构化输出
    res.save_to_img("./output/") ## 保存结果可视化图像
    res.save_to_json("./output/") ## 保存预测的结构化输出
```
得到的结果与命令行方式相同。

在上述 Python 脚本中，执行了如下几个步骤：

（1）实例化 `create_pipeline` 实例化产线对象：具体参数说明如下：

|参数|参数说明|参数类型|默认值|
|-|-|-|-|
|`pipeline`|产线名称或是产线配置文件路径。如为产线名称，则必须为 PaddleX 所支持的产线。|`str`|无|
|`device`|产线模型推理设备。支持：“gpu”，“cpu”。|`str`|`gpu`|
|`use_hpip`|是否启用高性能推理，仅当该产线支持高性能推理时可用。|`bool`|`False`|

（2）调用车辆属性识别产线对象的 `predict` 方法进行推理预测：`predict` 方法参数为`x`，用于输入待预测数据，支持多种输入方式，具体示例如下：

| 参数类型      | 参数说明                                                                                                  |
|---------------|-----------------------------------------------------------------------------------------------------------|
| Python Var    | 支持直接传入Python变量，如numpy.ndarray表示的图像数据。                                               |
| str         | 支持传入待预测数据文件路径，如图像文件的本地路径：`/root/data/img.jpg`。                                   |
| str           | 支持传入待预测数据文件URL，如图像文件的网络URL：[示例](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/vehicle_attribute_002.jpg)。|
| str           | 支持传入本地目录，该目录下需包含待预测数据文件，如本地路径：`/root/data/`。                               |
| dict          | 支持传入字典类型，字典的key需与具体任务对应，如车辆属性识别任务对应\"img\"，字典的val支持上述类型数据，例如：`{\"img\": \"/root/data1\"}`。|
| list          | 支持传入列表，列表元素需为上述类型数据，如`[numpy.ndarray, numpy.ndarray]，[\"/root/data/img1.jpg\", \"/root/data/img2.jpg\"]`，`[\"/root/data1\", \"/root/data2\"]`，`[{\"img\": \"/root/data1\"}, {\"img\": \"/root/data2/img.jpg\"}]`。|

（3）调用`predict`方法获取预测结果：`predict` 方法为`generator`，因此需要通过调用获得预测结果，`predict`方法以batch为单位对数据进行预测，因此预测结果为list形式表示的一组预测结果。

（4）对预测结果进行处理：每个样本的预测结果均为`dict`类型，且支持打印，或保存为文件，支持保存的类型与具体产线相关，如：

| 方法         | 说明                        | 方法参数                                                                                               |
|--------------|-----------------------------|--------------------------------------------------------------------------------------------------------|
| print        | 打印结果到终端              | `- format_json`：bool类型，是否对输出内容进行使用json缩进格式化，默认为True；<br>`- indent`：int类型，json格式化设置，仅当format_json为True时有效，默认为4；<br>`- ensure_ascii`：bool类型，json格式化设置，仅当format_json为True时有效，默认为False； |
| save_to_json | 将结果保存为json格式的文件   | `- save_path`：str类型，保存的文件路径，当为目录时，保存文件命名与输入文件类型命名一致；<br>`- indent`：int类型，json格式化设置，默认为4；<br>`- ensure_ascii`：bool类型，json格式化设置，默认为False； |
| save_to_img  | 将结果保存为图像格式的文件  | `- save_path`：str类型，保存的文件路径，当为目录时，保存文件命名与输入文件类型命名一致； |

若您获取了配置文件，即可对车辆属性识别产线各项配置进行自定义，只需要修改 `create_pipeline` 方法中的 `pipeline` 参数值为产线配置文件路径即可。

例如，若您的配置文件保存在 `./my_path/vehicle_attribute*.yaml` ，则只需执行：

```
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="./my_path/vehicle_attribute.yaml")
output = pipeline.predict("vehicle_attribute_002.jpg")
for res in output:
    res.print() ## 打印预测的结构化输出
    res.save_to_img("./output/") ## 保存结果可视化图像
    res.save_to_json("./output/") ## 保存预测的结构化输出
```
## 3. 开发集成/部署
如果产线可以达到您对产线推理速度和精度的要求，您可以直接进行开发集成/部署。

若您需要将产线直接应用在您的Python项目中，可以参考 [2.2.2 Python脚本方式](#222-python脚本方式集成)中的示例代码。

此外，PaddleX 也提供了其他三种部署方式，详细说明如下：

🚀 **高性能推理**：在实际生产环境中，许多应用对部署策略的性能指标（尤其是响应速度）有着较严苛的标准，以确保系统的高效运行与用户体验的流畅性。为此，PaddleX 提供高性能推理插件，旨在对模型推理及前后处理进行深度性能优化，实现端到端流程的显著提速，详细的高性能推理流程请参考[PaddleX高性能推理指南](../../../pipeline_deploy/high_performance_inference.md)。

☁️ **服务化部署**：服务化部署是实际生产环境中常见的一种部署形式。通过将推理功能封装为服务，客户端可以通过网络请求来访问这些服务，以获取推理结果。PaddleX 支持用户以低成本实现产线的服务化部署，详细的服务化部署流程请参考[PaddleX服务化部署指南](../../../pipeline_deploy/service_deploy.md)。

📱 **端侧部署**：端侧部署是一种将计算和数据处理功能放在用户设备本身上的方式，设备可以直接处理数据，而不需要依赖远程的服务器。PaddleX 支持将模型部署在 Android 等端侧设备上，详细的端侧部署流程请参考[PaddleX端侧部署指南](../../../pipeline_deploy/edge_deploy.md)。
您可以根据需要选择合适的方式部署模型产线，进而进行后续的 AI 应用集成。

## 4. 二次开发
如果车辆属性识别产线提供的默认模型权重在您的场景中，精度或速度不满意，您可以尝试利用**您自己拥有的特定领域或应用场景的数据**对现有模型进行进一步的**微调**，以提升车辆属性识别产线的在您的场景中的识别效果。

### 4.1 模型微调
由于车辆属性识别产线包含车辆属性识别模块和车辆检测模块，如果模型产线的效果不及预期可能来自于其中任何一个模块。
您可以对识别效果差的图片进行分析，如果在分析过程中发现有较多的主体目标未被检测出来，那么可能是车辆检测模型存在不足那么您需要参考[车辆检测模块开发教程](../../../module_usage/tutorials/cv_modules/human_detection.md)中的[二次开发](../../../module_usage/tutorials/cv_modules/human_detection.md#四二次开发)章节，使用您的私有数据集对车辆检测模型进行微调；如果检测出来的主体属性识别错误，那么您需要参考[车辆属性识别模块开发教程](../../../module_usage/tutorials/cv_modules/vehicle_attribute_recognition.md)中的[二次开发](../../../module_usage/tutorials/cv_modules/vehicle_attribute_recognition.md#四二次开发)章节，使用您的私有数据集对车辆属性识别模型进行微调。

### 4.2 模型应用
当您使用私有数据集完成微调训练后，可获得本地模型权重文件。

若您需要使用微调后的模型权重，只需对产线配置文件做修改，将微调后模型权重的本地路径替换至产线配置文件中的对应位置即可：

```
......
Pipeline:
  model: PP-LCNet_x1_0  #可修改为微调后模型的本地路径
  device: "gpu"
  batch_size: 1
......
```
随后， 参考本地体验中的命令行方式或 Python 脚本方式，加载修改后的产线配置文件即可。

##  5. 多硬件支持
PaddleX 支持英伟达 GPU、昆仑芯 XPU、昇腾 NPU和寒武纪 MLU 等多种主流硬件设备，**仅需修改 `--device` 参数**即可完成不同硬件之间的无缝切换。

例如，您使用英伟达 GPU 进行车辆属性识别产线的推理，使用的命令为：

```bash
paddlex --pipeline vehicle_attribute --input vehicle_attribute_002.jpg --device gpu:0
```
此时，若您想将硬件切换为昇腾 NPU，仅需将 `--device` 修改为 npu:0 即可：

```bash
paddlex --pipeline vehicle_attribute --input vehicle_attribute_002.jpg --device npu:0
```
若您想在更多种类的硬件上使用车辆属性识别产线，请参考[PaddleX多硬件使用指南](../../../other_devices_support/multi_devices_use_guide.md)。
