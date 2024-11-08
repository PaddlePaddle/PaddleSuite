---
comments: true
---

# 文本识别模块使用教程

## 一、概述
文本识别模块是OCR（光学字符识别）系统中的核心部分，负责从图像中的文本区域提取出文本信息。该模块的性能直接影响到整个OCR系统的准确性和效率。文本识别模块通常接收文本检测模块输出的文本区域的边界框（Bounding Boxes）作为输入，然后通过复杂的图像处理和深度学习算法，将图像中的文本转化为可编辑和可搜索的电子文本。文本识别结果的准确性，对于后续的信息提取和数据挖掘等应用至关重要。

## 二、支持模型列表

<table >
    <tr>
        <th>模型</th>
        <th>识别 Avg Accuracy(%)</th>
        <th>GPU推理耗时（ms）</th>
        <th>CPU推理耗时 (ms)</th>
        <th>模型存储大小（M）</th>
        <th>介绍</th>
    </tr>
    <tr>
        <td>PP-OCRv4_mobile_rec</td>
        <td>78.20</td>
        <td>7.95018</td>
        <td>46.7868</td>
        <td>10.6 M</td>
        <td rowspan="2">PP-OCRv4是百度飞桨视觉团队自研的文本识别模型PP-OCRv3的下一个版本，通过引入数据增强方案、GTC-NRTR指导分支等策略，在模型推理速度不变的情况下，进一步提升了文本识别精度。该模型提供了服务端（server）和移动端（mobile）两个不同版本，来满足不同场景下的工业需求。</td>
    </tr>
    <tr>
        <td>PP-OCRv4_server_rec </td>
        <td>79.20</td>
        <td>7.19439</td>
        <td>140.179</td>
        <td>71.2 M</td>
    </tr>
</table>

<b>注：以上精度指标的评估集是 PaddleOCR 自建的中文数据集，覆盖街景、网图、文档、手写多个场景，其中文本识别包含 1.1w 张图片。所有模型 GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为8，精度类型为 FP32。</b>

> ❗ 以上列出的是目标检测模块重点支持的<b>2个核心模型</b>，该模块总共支持<b>4个模型</b>，完整的模型列表如下：

<details><summary> 👉模型列表详情</summary>

<table >
<tr>
<th>模型</th>
<th>识别 Avg Accuracy(%)</th>
<th>GPU推理耗时（ms）</th>
<th>CPU推理耗时 (ms)</th>
<th>模型存储大小（M）</th>
<th>介绍</th>
</tr>
<tr>
<td>PP-OCRv4_mobile_rec</td>
<td>78.20</td>
<td>7.95018</td>
<td>46.7868</td>
<td>10.6 M</td>
<td rowspan="2">PP-OCRv4是百度飞桨视觉团队自研的文本识别模型PP-OCRv3的下一个版本，通过引入数据增强方案、GTC-NRTR指导分支等策略，在模型推理速度不变的情况下，进一步提升了文本识别精度。该模型提供了服务端（server）和移动端（mobile）两个不同版本，来满足不同场景下的工业需求。</td>
</tr>
<tr>
<td>PP-OCRv4_server_rec </td>
<td>79.20</td>
<td>7.19439</td>
<td>140.179</td>
<td>71.2 M</td>
</tr>
</table>

<p><b>注：以上精度指标的评估集是 PaddleOCR 自建的中文数据集，覆盖街景、网图、文档、手写多个场景，其中文本识别包含 1.1w 张图片。所有模型 GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为8，精度类型为 FP32。</b></p>
<table >
<tr>
<th>模型</th>
<th>识别 Avg Accuracy(%)</th>
<th>GPU推理耗时（ms）</th>
<th>CPU推理耗时</th>
<th>模型存储大小（M）</th>
<th>介绍</th>
</tr>
<tr>
<td>ch_SVTRv2_rec</td>
<td>68.81</td>
<td>8.36801</td>
<td>165.706</td>
<td>73.9 M</td>
<td rowspan="1">
SVTRv2 是一种由复旦大学视觉与学习实验室（FVL）的OpenOCR团队研发的服务端文本识别模型，其在PaddleOCR算法模型挑战赛 - 赛题一：OCR端到端识别任务中荣获一等奖，A榜端到端识别精度相比PP-OCRv4提升6%。
</td>
</tr>
</table>

<p><b>注：以上精度指标的评估集是 <a href="https://aistudio.baidu.com/competition/detail/1131/0/introduction">PaddleOCR算法模型挑战赛 - 赛题一：OCR端到端识别任务</a>A榜。 所有模型 GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为8，精度类型为 FP32。</b></p>
<table >
<tr>
<th>模型</th>
<th>识别 Avg Accuracy(%)</th>
<th>GPU推理耗时（ms）</th>
<th>CPU推理耗时</th>
<th>模型存储大小（M）</th>
<th>介绍</th>
</tr>
<tr>
<td>ch_RepSVTR_rec</td>
<td>65.07</td>
<td>10.5047</td>
<td>51.5647</td>
<td>22.1 M</td>
<td rowspan="1">    RepSVTR 文本识别模型是一种基于SVTRv2 的移动端文本识别模型，其在PaddleOCR算法模型挑战赛 - 赛题一：OCR端到端识别任务中荣获一等奖，B榜端到端识别精度相比PP-OCRv4提升2.5%，推理速度持平。</td>
</tr>
</table>

<p><b>注：以上精度指标的评估集是 <a href="https://aistudio.baidu.com/competition/detail/1131/0/introduction">PaddleOCR算法模型挑战赛 - 赛题一：OCR端到端识别任务</a>B榜。 所有模型 GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为8，精度类型为 FP32。</b></p></details>

## 三、快速集成
在快速集成前，首先需要安装 PaddleX 的 wheel 包，wheel的安装方式请参考[PaddleX本地安装教程](../../../installation/installation.md)。完成 wheel 包的安装后，几行代码即可完成文本识别模块的推理，可以任意切换该模块下的模型，您也可以将文本识别的模块中的模型推理集成到您的项目中。运行以下代码前，请您下载[示例图片](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_rec_001.png)到本地。

```python
from paddlex import create_model
model = create_model("PP-OCRv4_mobile_rec")
output = model.predict("general_ocr_rec_001.png", batch_size=1)
for res in output:
    res.print(json_format=False)
    res.save_to_img("./output/")
    res.save_to_json("./output/res.json")
```
关于更多 PaddleX 的单模型推理的 API 的使用方法，可以参考[PaddleX单模型Python脚本使用说明](../../instructions/model_python_API.md)。

## 四、二次开发
如果你追求更高精度的现有模型，可以使用 PaddleX 的二次开发能力，开发更好的文本识别模型。在使用 PaddleX 开发文本识别模型之前，请务必安装 PaddleX 的 OCR 相关模型训练插件，安装过程可以参考[PaddleX本地安装教程](../../../installation/installation.md)中的二次开发部分。

### 4.1 数据准备
在进行模型训练前，需要准备相应任务模块的数据集。PaddleX 针对每一个模块提供了数据校验功能，<b>只有通过数据校验的数据才可以进行模型训练</b>。此外，PaddleX 为每一个模块都提供了 Demo 数据集，您可以基于官方提供的 Demo 数据完成后续的开发。若您希望用私有数据集进行后续的模型训练，可以参考[PaddleX文本检测/文本识别任务模块数据标注教程](../../../data_annotations/ocr_modules/text_detection_recognition.md)。

#### 4.1.1 Demo 数据下载
您可以参考下面的命令将 Demo 数据集下载到指定文件夹：

```bash
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/ocr_rec_dataset_examples.tar -P ./dataset
tar -xf ./dataset/ocr_rec_dataset_examples.tar -C ./dataset/
```
#### 4.1.2 数据校验
一行命令即可完成数据校验：

```bash
python main.py -c paddlex/configs/text_recognition/PP-OCRv4_mobile_rec.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/ocr_rec_dataset_examples
```
执行上述命令后，PaddleX 会对数据集进行校验，并统计数据集的基本信息，命令运行成功后会在log中打印出`Check dataset passed !`信息。校验结果文件保存在`./output/check_dataset_result.json`，同时相关产出会保存在当前目录的`./output/check_dataset`目录下，产出目录中包括可视化的示例样本图片和样本分布直方图。

<details><summary>👉 <b>校验结果详情（点击展开）</b></summary>

<p>校验结果文件具体内容为：</p>
<pre><code class="language-bash">{
  &quot;done_flag&quot;: true,
  &quot;check_pass&quot;: true,
  &quot;attributes&quot;: {
    &quot;train_samples&quot;: 4468,
    &quot;train_sample_paths&quot;: [
      &quot;../dataset/ocr_rec_dataset_examples/images/train_word_1.png&quot;,
      &quot;../dataset/ocr_rec_dataset_examples/images/train_word_10.png&quot;
    ],
    &quot;val_samples&quot;: 2077,
    &quot;val_sample_paths&quot;: [
      &quot;../dataset/ocr_rec_dataset_examples/images/val_word_1.png&quot;,
      &quot;../dataset/ocr_rec_dataset_examples/images/val_word_10.png&quot;
    ]
  },
  &quot;analysis&quot;: {
    &quot;histogram&quot;: &quot;check_dataset/histogram.png&quot;
  },
  &quot;dataset_path&quot;: &quot;./dataset/ocr_rec_dataset_examples&quot;,
  &quot;show_type&quot;: &quot;image&quot;,
  &quot;dataset_type&quot;: &quot;MSTextRecDataset&quot;
}
</code></pre>
<p>上述校验结果中，<code>check_pass</code> 为 <code>true</code> 表示数据集格式符合要求，其他部分指标的说明如下：</p>
<ul>
<li><code>attributes.train_samples</code>：该数据集训练集样本数量为 4468；</li>
<li><code>attributes.val_samples</code>：该数据集验证集样本数量为 2077；</li>
<li><code>attributes.train_sample_paths</code>：该数据集训练集样本可视化图片相对路径列表；</li>
<li><code>attributes.val_sample_paths</code>：该数据集验证集样本可视化图片相对路径列表；
另外，数据集校验还对数据集中所有字符长度占比的分布情况进行了分析，并绘制了分布直方图（histogram.png）：</li>
</ul>
<p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/modules/text_recog/01.png"></p></details>

#### 4.1.3 数据集格式转换/数据集划分（可选）
在您完成数据校验之后，可以通过<b>修改配置文件</b>或是<b>追加超参数</b>的方式对数据集的格式进行转换，也可以对数据集的训练/验证比例进行重新划分。

<details><summary>👉 <b>格式转换/数据集划分详情（点击展开）</b></summary>

<p><b>（1）数据集格式转换</b></p>
<p>文本识别暂不支持数据转换。</p>
<p><b>（2）数据集划分</b></p>
<p>数据集划分的参数可以通过修改配置文件中 <code>CheckDataset</code> 下的字段进行设置，配置文件中部分参数的示例说明如下：</p>
<ul>
<li><code>CheckDataset</code>:</li>
<li><code>split</code>:</li>
<li><code>enable</code>: 是否进行重新划分数据集，为 <code>True</code> 时进行数据集格式转换，默认为 <code>False</code>；</li>
<li><code>train_percent</code>: 如果重新划分数据集，则需要设置训练集的百分比，类型为0-100之间的任意整数，需要保证和 <code>val_percent</code> 值加和为100；
例如，您想重新划分数据集为 训练集占比90%、验证集占比10%，则需将配置文件修改为：</li>
</ul>
<pre><code class="language-bash">......
CheckDataset:
  ......
  split:
    enable: True
    train_percent: 90
    val_percent: 10
  ......
</code></pre>
<p>随后执行命令：</p>
<pre><code class="language-bash">python main.py -c paddlex/configs/text_recognition/PP-OCRv4_mobile_rec.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/ocr_rec_dataset_examples
</code></pre>
<p>数据划分执行之后，原有标注文件会被在原路径下重命名为 <code>xxx.bak</code>。</p>
<p>以上参数同样支持通过追加命令行参数的方式进行设置：</p>
<pre><code class="language-bash">python main.py -c paddlex/configs/text_recognition/PP-OCRv4_mobile_rec.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/ocr_rec_dataset_examples \
    -o CheckDataset.split.enable=True \
    -o CheckDataset.split.train_percent=90 \
    -o CheckDataset.split.val_percent=10
</code></pre></details>

### 4.2 模型训练
一条命令即可完成模型的训练，以此处 PP-OCRv4 移动端文本识别模型（PP-OCRv4_mobile_rec）的训练为例：

```bash
python main.py -c paddlex/configs/text_recognition/PP-OCRv4_mobile_rec.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/ocr_rec_dataset_examples
```
需要如下几步：

* 指定模型的`.yaml` 配置文件路径（此处为`PP-OCRv4_mobile_rec.yaml`，训练其他模型时，需要的指定相应的配置文件，模型和配置的文件的对应关系，可以查阅[PaddleX模型列表（CPU/GPU）](../../../support_list/models_list.md)）
* 指定模式为模型训练：`-o Global.mode=train`
* 指定训练数据集路径：`-o Global.dataset_dir`
其他相关参数均可通过修改`.yaml`配置文件中的`Global`和`Train`下的字段来进行设置，也可以通过在命令行中追加参数来进行调整。如指定前 2 卡 gpu 训练：`-o Global.device=gpu:0,1`；设置训练轮次数为 10：`-o Train.epochs_iters=10`。更多可修改的参数及其详细解释，可以查阅模型对应任务模块的配置文件说明[PaddleX通用模型配置文件参数说明](../../instructions/config_parameters_common.md)。



<details><summary>👉 <b>更多说明（点击展开）</b></summary>

<ul>
<li>模型训练过程中，PaddleX 会自动保存模型权重文件，默认为<code>output</code>，如需指定保存路径，可通过配置文件中 <code>-o Global.output</code> 字段进行设置。</li>
<li>PaddleX 对您屏蔽了动态图权重和静态图权重的概念。在模型训练的过程中，会同时产出动态图和静态图的权重，在模型推理时，默认选择静态图权重推理。</li>
<li>
<p>在完成模型训练后，所有产出保存在指定的输出目录（默认为<code>./output/</code>）下，通常有以下产出：</p>
</li>
<li>
<p><code>train_result.json</code>：训练结果记录文件，记录了训练任务是否正常完成，以及产出的权重指标、相关文件路径等；</p>
</li>
<li><code>train.log</code>：训练日志文件，记录了训练过程中的模型指标变化、loss 变化等；</li>
<li><code>config.yaml</code>：训练配置文件，记录了本次训练的超参数的配置；</li>
<li><code>.pdparams</code>、<code>.pdema</code>、<code>.pdopt.pdstate</code>、<code>.pdiparams</code>、<code>.pdmodel</code>：模型权重相关文件，包括网络参数、优化器、EMA、静态图网络参数、静态图网络结构等；</li>
</ul></details>

## <b>4.3 模型评估</b>
在完成模型训练后，可以对指定的模型权重文件在验证集上进行评估，验证模型精度。使用 PaddleX 进行模型评估，一条命令即可完成模型的评估：

```bash
python main.py -c paddlex/configs/text_recognition/PP-OCRv4_mobile_rec.yaml \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/ocr_rec_dataset_examples

```
与模型训练类似，需要如下几步：

* 指定模型的`.yaml` 配置文件路径（此处为`PP-OCRv4_mobile_rec.yaml`）
* 指定模式为模型评估：`-o Global.mode=evaluate`
* 指定验证数据集路径：`-o Global.dataset_dir`
其他相关参数均可通过修改`.yaml`配置文件中的`Global`和`Evaluate`下的字段来进行设置，详细请参考[PaddleX通用模型配置文件参数说明](../../instructions/config_parameters_common.md)。

<details><summary>👉 <b>更多说明（点击展开）</b></summary>

<p>在模型评估时，需要指定模型权重文件路径，每个配置文件中都内置了默认的权重保存路径，如需要改变，只需要通过追加命令行参数的形式进行设置即可，如<code>-o Evaluate.weight_path=./output/best_accuracy/best_accuracy.pdparams</code>。</p>
<p>在完成模型评估后，通常有以下产出：<code>evaluate_result.json，其记录了</code>评估的结果，具体来说，记录了评估任务是否正常完成，以及模型的评估指标，包含 acc、norm_edit_dis；</p></details>

### <b>4.4 模型推理和模型集成</b>
在完成模型的训练和评估后，即可使用训练好的模型权重进行推理预测或者进行Python集成。

#### 4.4.1 模型推理
通过命令行的方式进行推理预测，只需如下一条命令。运行以下代码前，请您下载[示例图片](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_rec_001.png)到本地。

```bash
python main.py -c paddlex/configs/text_recognition/PP-OCRv4_mobile_rec.yaml \
    -o Global.mode=predict \
    -o Predict.model_dir="./output/best_accuracy/inference" \
    -o Predict.input="general_ocr_rec_001.png"
```
与模型训练和评估类似，需要如下几步：

* 指定模型的`.yaml` 配置文件路径（此处为`PP-OCRv4_mobile_rec.yaml`）
* 指定模式为模型推理预测：`-o Global.mode=predict`
* 指定模型权重路径：`-o Predict.model_dir="./output/best_accuracy/inference"`
* 指定输入数据路径：`-o Predict.input="..."`
其他相关参数均可通过修改`.yaml`配置文件中的`Global`和`Predict`下的字段来进行设置，详细请参考[PaddleX通用模型配置文件参数说明](../../instructions/config_parameters_common.md)。

#### 4.4.2 模型集成
模型可以直接集成到 PaddleX 产线中，也可以直接集成到您自己的项目中。

1.<b>产线集成</b>

文本识别模块可以集成的PaddleX产线有[通用 OCR 产线](../../../pipeline_usage/tutorials/ocr_pipelines/OCR.md)、[通用表格识别产线](../../../pipeline_usage/tutorials/ocr_pipelines/table_recognition.md)、[文档场景信息抽取v3产线（PP-ChatOCRv3）](../../../pipeline_usage/tutorials/information_extraction_pipelines/document_scene_information_extraction.md)，只需要替换模型路径即可完成相关产线的文本识别模块的模型更新。

2.<b>模块集成</b>

您产出的权重可以直接集成到文本识别模块中，可以参考[快速集成](#三快速集成)的 Python 示例代码，只需要将模型替换为你训练的到的模型路径即可。
