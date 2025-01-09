---
comments: true
---

# 旋转目标检测模块使用教程

## 一、概述
旋转目标检测是目标检测模块中的一种衍生，它专门针对旋转目标进行检测。旋转框（Rotated Bounding Boxes）常用于检测带有角度信息的矩形框，即矩形框的宽和高不再与图像坐标轴平行。相较于水平矩形框，旋转矩形框一般包括更少的背景信息。旋转框检测常用于遥感等场景中。

## 二、支持模型列表

<table>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>mAP(%)</th>
<th>GPU推理耗时 (ms)</th>
<th>CPU推理耗时 (ms)</th>
<th>模型存储大小 (M)</th>
<th>介绍</th>
</tr>
<tr>
<td>PP-YOLOE-R_L</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b1_v2/PP-YOLOE-R_L_infer.tar">推理模型</a>/<a href="https://paddledet.bj.bcebos.com/models/ppyoloe_r_crn_l_3x_dota.pdparams">训练模型</a></td>
<td>78.14</td>
<td>20.7039</td>
<td>157.942</td>
<td>211.0 M</td>
<td rowspan="1">PP-YOLOE-R是一个高效的单阶段Anchor-free旋转框检测模型。基于PP-YOLOE, PP-YOLOE-R以极少的参数量和计算量为代价，引入了一系列有用的设计来提升检测精度。</td>
</tr>
</table>
<p><b>注：以上精度指标为<a href="https://captain-whu.github.io/DOTA/">DOTA</a>验证集 mAP(0.5:0.95)。所有模型 GPU 推理耗时基于 NVIDIA TRX2080 Ti 机器，精度类型为 F16， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为8，精度类型为 FP32。</b></p>
> ❗ 以上列出的是paddleX当前支持的旋转目标检测模型</b>，实际的PaddleDetection套件支持<b>10</b>个旋转目标检测模型，详细模型列表请参考<a href="https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.8/configs/rotate">PaddleDetection</a>


## 三、快速集成
> ❗ 在快速集成前，请先安装 PaddleX 的 wheel 包，详细请参考 [PaddleX本地安装教程](../../../installation/installation.md)

完成 wheel 包的安装后，几行代码即可完成旋转目标检测模块的推理，可以任意切换该模块下的模型，您也可以将旋转目标检测的模块中的模型推理集成到您的项目中。运行以下代码前，请您下载[示例图片](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/rotated_object_detection_001.png)到本地。

```python
from paddlex import create_model
model = create_model("PP-YOLOE-R_L")
output = model.predict("rotated_object_detection_001.png", batch_size=1)
for res in output:
    res.print(json_format=False)
    res.save_to_img("./output/")
    res.save_to_json("./output/res.json")
```
关于更多 PaddleX 的单模型推理的 API 的使用方法，可以参考[PaddleX单模型Python脚本使用说明](../../instructions/model_python_API.md)。

## 四、二次开发
如果你追求更高精度的现有模型，可以使用 PaddleX 的二次开发能力，开发更好的旋转目标检测模型。在使用 PaddleX 开发旋转目标检测模型之前，请务必安装 PaddleX的旋转目标检测相关模型训练插件，安装过程可以参考 [PaddleX本地安装教程](../../../installation/installation.md)

### 4.1 数据准备
在进行模型训练前，需要准备相应任务模块的数据集。PaddleX 针对每一个模块提供了数据校验功能，<b>只有通过数据校验的数据才可以进行模型训练</b>。此外，PaddleX 为每一个模块都提供了 Demo 数据集，您可以基于官方提供的 Demo 数据完成后续的开发。若您希望用私有数据集进行后续的模型训练，可以参考[PaddleX目标检测任务模块数据标注教程](../../../data_annotations/cv_modules/object_detection.md)。

#### 4.1.1 Demo 数据下载
您可以参考下面的命令将 Demo 数据集下载到指定文件夹：

```bash
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/rdet_dota_examples.tar -P ./dataset
tar -xf ./dataset/rdet_dota_examples.tar -C ./dataset/
```
解压后，数据集目录结构如下：
```bash
- dataset/DOTA-sampled200_crop1024_data
  - annotations
    - instance_train.json
    - instance_val.json
  - images
    - img1.png
    - img2.png
    - img3.png
    ...
```
#### 4.1.2 数据校验
一行命令即可完成数据校验：

```bash
python main.py -c paddlex/configs/rotated_object_detection/PP-YOLOE-R_L.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/DOTA-sampled200_crop1024_data
```
执行上述命令后，PaddleX 会对数据集进行校验，并统计数据集的基本信息，命令运行成功后会在log中打印出`Check dataset passed !`信息。校验结果文件保存在`./output/check_dataset_result.json`，同时相关产出会保存在当前目录的`./output/check_dataset`目录下，产出目录中包括可视化的示例样本图片和样本分布直方图。

<details><summary>👉 <b>校验结果详情（点击展开）</b></summary>

<p>校验结果文件具体内容为：</p>
<pre><code class="language-bash">{
  &quot;done_flag&quot;: true,
  &quot;check_pass&quot;: true,
  &quot;attributes&quot;: {
    &quot;num_classes&quot;: 15,
    &quot;train_samples&quot;: 1892,
    &quot;train_sample_paths&quot;: [
      &quot;check_dataset\/demo_img\/P2610__1.0__0___0.png&quot;,
      &quot;check_dataset\/demo_img\/P1137__1.0__0___0.png&quot;,
      &quot;check_dataset\/demo_img\/P1122__1.0__5888___1648.png&quot;,
      &quot;check_dataset\/demo_img\/P0543__1.0__0___0.png&quot;,
      &quot;check_dataset\/demo_img\/P0518__1.0__0___91.png&quot;,
      &quot;check_dataset\/demo_img\/P0961__1.0__1648___87.png&quot;,
      &quot;check_dataset\/demo_img\/P1732__1.0__0___824.png&quot;,
      &quot;check_dataset\/demo_img\/P2766__1.0__4421___0.png&quot;,
      &quot;check_dataset\/demo_img\/P2582__1.0__674___725.png&quot;,
      &quot;check_dataset\/demo_img\/P1529__1.0__2976___1648.png&quot;
    ],
    &quot;val_samples&quot;: 473,
    &quot;val_sample_paths&quot;: [
      &quot;check_dataset\/demo_img\/P2342__1.0__890___0.png&quot;,
      &quot;check_dataset\/demo_img\/P1386__1.0__2472___1648.png&quot;,
      &quot;check_dataset\/demo_img\/P0961__1.0__824___87.png&quot;,
      &quot;check_dataset\/demo_img\/P1651__1.0__824___824.png&quot;,
      &quot;check_dataset\/demo_img\/P1529__1.0__824___2976.png&quot;,
      &quot;check_dataset\/demo_img\/P0961__1.0__4944___87.png&quot;,
      &quot;check_dataset\/demo_img\/P0725__1.0__634___0.png&quot;,
      &quot;check_dataset\/demo_img\/P1679__1.0__1648___1648.png&quot;,
      &quot;check_dataset\/demo_img\/P2726__1.0__824___1578.png&quot;,
      &quot;check_dataset\/demo_img\/P0457__1.0__379___0.png&quot;,
    ]
  },
  &quot;analysis&quot;: {
    &quot;histogram&quot;: &quot;check_dataset/histogram.png&quot;
  },
  &quot;dataset_path&quot;: &quot;./dataset/DOTA-sampled200_crop1024_data&quot;,
  &quot;show_type&quot;: &quot;image&quot;,
  &quot;dataset_type&quot;: &quot;COCODetDataset&quot;
}
</code></pre>
<p>上述校验结果中，check_pass 为 true 表示数据集格式符合要求，其他部分指标的说明如下：</p>
<ul>
<li><code>attributes.num_classes</code>：该数据集类别数为 15；</li>
<li><code>attributes.train_samples</code>：该数据集训练集样本数量为 1892；</li>
<li><code>attributes.val_samples</code>：该数据集验证集样本数量为 473；</li>
<li><code>attributes.train_sample_paths</code>：该数据集训练集样本可视化图片相对路径列表；</li>
<li><code>attributes.val_sample_paths</code>：该数据集验证集样本可视化图片相对路径列表；</li>
</ul>
<p>另外，数据集校验还对数据集中所有类别的样本数量分布情况进行了分析，并绘制了分布直方图（histogram.png）：</p>
<p><img src="https://raw.githubusercontent.com/BluebirdStory/PaddleX_doc_images/main/images/modules/robj_det/01.png"></p></details>

#### 4.1.3 数据集格式转换/数据集划分（可选）
在您完成数据校验之后，可以通过<b>修改配置文件</b>或是<b>追加超参数</b>的方式对数据集的格式进行转换，也可以对数据集的训练/验证比例进行重新划分。

<details><summary>👉 <b>格式转换/数据集划分详情（点击展开）</b></summary>

<p><b>（1）数据集格式转换</b></p>

旋转目标检测赞不支持数据格式转换，只支持标准DOTA的COCO数据格式。

<p><b>（2）数据集划分</b></p>
<p>数据集划分的参数可以通过修改配置文件中 <code>CheckDataset</code> 下的字段进行设置，配置文件中部分参数的示例说明如下：</p>
<ul>
<li><code>CheckDataset</code>:</li>
<li><code>split</code>:</li>
<li><code>enable</code>: 是否进行重新划分数据集，为 <code>True</code> 时进行数据集格式转换，默认为 <code>False</code>；</li>
<li><code>train_percent</code>: 如果重新划分数据集，则需要设置训练集的百分比，类型为0-100之间的任意整数，需要保证和 <code>val_percent</code> 值加和为100；</li>
<li><code>val_percent</code>: 如果重新划分数据集，则需要设置验证集的百分比，类型为0-100之间的任意整数，需要保证和 <code>train_percent</code> 值加和为100；
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
<pre><code class="language-bash">python main.py -c paddlex/configs/rotated_object_detection/PP-YOLOE-R_L.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/DOTA-sampled200_crop1024_data
</code></pre>
<p>数据划分执行之后，原有标注文件会被在原路径下重命名为 <code>xxx.bak</code>。</p>
<p>以上参数同样支持通过追加命令行参数的方式进行设置：</p>
<pre><code class="language-bash">python main.py -c paddlex/configs/rotated_object_detection/PP-YOLOE-R_L.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/DOTA-sampled200_crop1024_data \
    -o CheckDataset.split.enable=True \
    -o CheckDataset.split.train_percent=90 \
    -o CheckDataset.split.val_percent=10
</code></pre></details>

### 4.2 模型训练
一条命令即可完成模型的训练，以此处旋转目标检测模型 `PP-YOLOE-R_L` 的训练为例：

```bash
python main.py -c paddlex/configs/rotated_object_detection/PP-YOLOE-R_L.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/DOTA-sampled200_crop1024_data
```
需要如下几步：

* 指定模型的`.yaml` 配置文件路径（此处为`PP-YOLOE-R_L.yaml`，训练其他模型时，需要的指定相应的配置文件，模型和配置的文件的对应关系，可以查阅[PaddleX模型列表（CPU/GPU）](../../../support_list/models_list.md)）
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
python main.py -c paddlex/configs/rotated_object_detection/PP-YOLOE-R_L.yaml \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/DOTA-sampled200_crop1024_data
```
与模型训练类似，需要如下几步：

* 指定模型的`.yaml` 配置文件路径（此处为`PP-YOLOE-R_L.yaml`）
* 指定模式为模型评估：`-o Global.mode=evaluate`
* 指定验证数据集路径：`-o Global.dataset_dir`
其他相关参数均可通过修改`.yaml`配置文件中的`Global`和`Evaluate`下的字段来进行设置，详细请参考[PaddleX通用模型配置文件参数说明](../../instructions/config_parameters_common.md)。

<details><summary>👉 <b>更多说明（点击展开）</b></summary>

<p>在模型评估时，需要指定模型权重文件路径，每个配置文件中都内置了默认的权重保存路径，如需要改变，只需要通过追加命令行参数的形式进行设置即可，如<code>-o Evaluate.weight_path=./output/best_model/best_model.pdparams</code>。</p>
<p>在完成模型评估后，会产出<code>evaluate_result.json，其记录了</code>评估的结果，具体来说，记录了评估任务是否正常完成，以及模型的评估指标，包含 AP；</p></details>

### <b>4.4 模型推理和模型集成</b>
在完成模型的训练和评估后，即可使用训练好的模型权重进行推理预测或者进行Python集成。

#### 4.4.1 模型推理

* 通过命令行的方式进行推理预测，只需如下一条命令。运行以下代码前，请您下载[示例图片](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/rotated_object_detection_001.png)到本地。
```bash
python main.py -c paddlex/configs/rotated_object_detection/PP-YOLOE-R_L.yaml  \
    -o Global.mode=predict \
    -o Predict.model_dir="./output/best_model/inference" \
    -o Predict.input="rotated_object_detection_001.png"
```
与模型训练和评估类似，需要如下几步：

* 指定模型的`.yaml` 配置文件路径（此处为`PP-YOLOE-R_L.yaml`）
* 指定模式为模型推理预测：`-o Global.mode=predict`
* 指定模型权重路径：`-o Predict.model_dir="./output/best_model/inference"`
* 指定输入数据路径：`-o Predict.input="..."`
其他相关参数均可通过修改`.yaml`配置文件中的`Global`和`Predict`下的字段来进行设置，详细请参考[PaddleX通用模型配置文件参数说明](../../instructions/config_parameters_common.md)。

#### 4.4.2 模型集成
模型可以直接集成到 PaddleX 产线中，也可以直接集成到您自己的项目中。

1.<b>模块集成</b>

您产出的权重可以直接集成到旋转目标检测模块中，可以参考[快速集成](#三快速集成)的 Python 示例代码，只需要将模型替换为你训练的到的模型路径即可。
