---
comments: true
---

# 多语种语音识别模块使用教程

## 一、概述
语音识别是一种先进的工具，它能够将人类口述的多种语言自动转换为相应的文本或命令。该技术还在智能客服、语音助手、会议记录等多个领域发挥着重要作用。多语种语音识别则可以支持自动进行语种检索，支持多种不同语言的语音的识别。

## 二、支持模型列表

### Whisper Model
Demo Link | Training Data | Size | Descriptions | CER | Model
:-----------: | :-----:| :-------: | :-----: | :-----: |:---------:|
Whisper | 680kh from internet | large: 5.8G,</br>medium: 2.9G,</br>small: 923M,</br>base: 277M,</br>tiny: 145M | Encoder:Transformer,</br> Decoder:Transformer, </br>Decoding method: </br>Greedy search | 0.027 </br>(large, Librispeech) | [whisper-large](https://paddlespeech.bj.bcebos.com/whisper/whisper_model_20221122/whisper-large-model.tar.gz) </br>[whisper-medium](https://paddlespeech.bj.bcebos.com/whisper/whisper_model_20221122/whisper-medium-model.tar.gz) </br>[whisper-small](https://paddlespeech.bj.bcebos.com/whisper/whisper_model_20221122/whisper-small-model.tar.gz) </br>[whisper-base](https://paddlespeech.bj.bcebos.com/whisper/whisper_model_20221122/whisper-base-model.tar.gz) </br>[whisper-tiny](https://paddlespeech.bj.bcebos.com/whisper/whisper_model_20221122/whisper-tiny-model.tar.gz) </br>

## 三、快速集成
在快速集成前，首先需要安装 PaddleX 的 wheel 包，wheel的安装方式请参考[PaddleX本地安装教程](../../../installation/installation.md)。完成 wheel 包的安装后，几行代码即可完成文本识别模块的推理，可以任意切换该模块下的模型，您也可以将文本识别的模块中的模型推理集成到您的项目中。运行以下代码前，请您下载[示例语音](https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav)到本地。

```python
from paddlex import create_model
model = create_model("whisper_large")
output = model.predict("./zh.wav", batch_size=1)
for res in output:
    res.print(json_format=False)
```
关于更多 PaddleX 的单模型推理的 API 的使用方法，可以参考[PaddleX单模型Python脚本使用说明](../../instructions/model_python_API.md)。

## 四、二次开发
目前该模型仅支持推理，暂不支持模型的训练。

### 4.1 数据准备
在进行模型训练前，需要准备相应任务模块的数据集。PaddleX 针对每一个模块提供了数据校验功能，<b>只有通过数据校验的数据才可以进行模型训练</b>。此外，PaddleX 为每一个模块都提供了 Demo 数据集，您可以基于官方提供的 Demo 数据完成后续的开发。

#### 4.1.1 Demo 数据下载
您可以参考下面的命令将 Demo 数据集下载到指定文件夹：

```bash
wget https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav
```

### 4.2 模型训练
暂不支持

## <b>4.3 模型评估</b>
暂不支持

### <b>4.4 模型推理和模型集成</b>

#### 4.4.1 模型推理
通过命令行的方式进行推理预测，只需如下一条命令。运行以下代码前，请您下载[示例语音](https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav)到本地。

```bash
python main.py -c paddlex/configs/modules/multilingual_speech_recognition/whisper_large.yaml \
    -o Global.mode=predict \
    -o Predict.input="./zh.wav"
```
模型推理配置需要如下几步：

* 指定模型的`.yaml` 配置文件路径（此处为`whisper_large.yaml`）
* 指定模式为模型推理预测：`-o Global.mode=predict`
* 指定输入数据路径：`-o Predict.input="..."`
其他相关参数均可通过修改`.yaml`配置文件中的`Global`和`Predict`下的字段来进行设置，详细请参考[PaddleX通用模型配置文件参数说明](../../instructions/config_parameters_common.md)。

#### 4.4.2 模型集成
模型可以直接集成到 PaddleX 产线中，也可以直接集成到您自己的项目中。

1.<b>产线集成</b>

暂无示例。

2.<b>模块集成</b>

您产出的权重可以直接集成到文本识别模块中，可以参考[快速集成](#三快速集成)的 Python 示例代码，只需要将模型替换为你训练的到的模型路径即可。
