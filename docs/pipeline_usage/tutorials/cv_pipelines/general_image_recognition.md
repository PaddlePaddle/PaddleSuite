---
comments: true
---

# 通用图像识别产线使用教程

## 1. 通用图像识别产线介绍

通用图像识别产线旨在解决开放域目标定位及识别问题，目前 PaddleX 的通用图像识别产线支持 PP-ShiTuV2。

PP-ShiTuV2 是一个实用的通用图像识别系统，主要由主体检测、特征学习和向量检索三个模块组成。该系统从骨干网络选择和调整、损失函数的选择、数据增强、学习率变换策略、正则化参数选择、预训练模型使用以及模型裁剪量化多个方面，融合改进多种策略，对各个模块进行优化，最终在多个实际应用场景上的检索性能均有较好效果。

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/pipelines/general_image_recognition/pp_shitu_v2.jpg">

<b>通用图像识别产线中包含了主体检测模块和图像特征模块</b>，有若干模型可供选择，您可以根据下边的 benchmark 数据来选择使用的模型。<b>如您更考虑模型精度，请选择精度较高的模型，如您更考虑模型推理速度，请选择推理速度较快的模型，如您更考虑模型存储大小，请选择存储大小较小的模型</b>。

<summary> 👉模型列表详情</summary>

<b>主体检测模块：</b>

<table>
<tr>
<th>模型</th>
<th>mAP(0.5:0.95)</th>
<th>mAP(0.5)</th>
<th>GPU推理耗时（ms）</th>
<th>CPU推理耗时 (ms)</th>
<th>模型存储大小（M）</th>
<th>介绍</th>
</tr>
<tr>
<td>PP-ShiTuV2_det</td>
<td>41.5</td>
<td>62.0</td>
<td>33.7</td>
<td>537.0</td>
<td>27.54</td>
<td>基于PicoDet_LCNet_x2_5的主体检测模型，模型可能会同时检测出多个常见主体。</td>
</tr>
</table>

注：以上精度指标为 PaddleClas 主体检测数据集。

<b>图像特征模块：</b>


<table>
<tr>
<th>模型</th>
<th>recall@1 (%)</th>
<th>GPU推理耗时 (ms)</th>
<th>CPU推理耗时 (ms)</th>
<th>模型存储大小 (M)</th>
<th>介绍</th>
</tr>
<tr>
<td>PP-ShiTuV2_rec</td>
<td>84.2</td>
<td>5.23428</td>
<td>19.6005</td>
<td>16.3 M</td>
<td rowspan="3">PP-ShiTuV2是一个通用图像特征系统，由主体检测、特征提取、向量检索三个模块构成，这些模型是其中的特征提取模块的模型之一，可以根据系统的情况选择不同的模型。</td>
</tr>
<tr>
<td>PP-ShiTuV2_rec_CLIP_vit_base</td>
<td>88.69</td>
<td>13.1957</td>
<td>285.493</td>
<td>306.6 M</td>
</tr>
<tr>
<td>PP-ShiTuV2_rec_CLIP_vit_large</td>
<td>91.03</td>
<td>51.1284</td>
<td>1131.28</td>
<td>1.05 G</td>
</tr>
</table>

注：以上精度指标为 AliProducts recall@1。所有模型 GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为8，精度类型为 FP32。

## 2. 快速开始

PaddleX 所提供的预训练的模型产线均可以快速体验效果，你可以在本地使用 Python 体验通用图像识别产线的效果。

### 2.1 在线体验

暂不支持在线体验。

### 2.2 本地体验

> ❗ 在本地使用通用图像识别产线前，请确保您已经按照[PaddleX安装教程](../../../installation/installation.md)完成了PaddleX的wheel包安装。

#### 2.2.1 命令行方式体验

该产线暂不支持命令行体验。

默认使用内置的的通用图像识别产线配置文件，若您需要自定义配置文件，可执行如下命令获取：

<details><summary> 👉点击展开</summary>

<pre><code class="language-bash">paddlex --get_pipeline_config PP-ShiTuV2
</code></pre>
<p>执行后，通用图像识别产线配置文件将被保存在当前路径。若您希望自定义保存位置，可执行如下命令（假设自定义保存位置为<code>./my_path</code>）：</p>
<pre><code class="language-bash">paddlex --get_pipeline_config PP-ShiTuV2 --save_path ./my_path
</code></pre></details>

#### 2.2.2 Python脚本方式集成

* 在该产线的运行示例中需要预先构建特征向量库，您可以下载官方提供的饮料识别测试数据集[drink_dataset_v2.0](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/data/drink_dataset_v2.0.tar) 构建特征向量库。若您希望用私有数据集，可以参考[2.3节 构建特征库的数据组织方式](#23-构建特征库的数据组织方式)。之后通过几行代码即可完成建立特征向量库和通用图像识别产线的快速推理。

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="PP-ShiTuV2")

pipeline.build_index(data_root="drink_dataset_v2.0/gallery/", index_dir="index_dir")

output = pipeline.predict("./drink_dataset_v2.0/test_images/", index_dir="index_dir")
for res in output:
    res.print()
    res.save_to_img("./output/")
```

在上述 Python 脚本中，执行了如下几个步骤：

（1）实例化 `create_pipeline` 实例化 通用图像识别 产线对象。具体参数说明如下：

<table>
<thead>
<tr>
<th>参数</th>
<th>参数说明</th>
<th>参数类型</th>
<th>默认值</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>pipeline</code></td>
<td>产线名称或是产线配置文件路径。如为产线名称，则必须为 PaddleX 所支持的产线。</td>
<td><code>str</code></td>
<td>无</td>
</tr>
<tr>
<td><code>index_dir</code></td>
<td>产线推理预测所用的检索库文件所在的目录，如不传入该参数，则需要在<code>predict()</code>中指定<code>index_dir</code>。</td>
<td><code>str</code></td>
<td>None</td>
</tr>
<tr>
<td><code>device</code></td>
<td>产线模型推理设备。支持：“gpu”，“cpu”。</td>
<td><code>str</code></td>
<td><code>gpu</code></td>
</tr>
<tr>
<td><code>use_hpip</code></td>
<td>是否启用高性能推理，仅当该产线支持高性能推理时可用。</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
</tbody>
</table>
（2）调用通用图像识别产线对象的 `build_index` 方法，构建特征向量库。具体参数为说明如下：

<table>
<thead>
<tr>
<th>参数</th>
<th>参数说明</th>
<th>参数类型</th>
<th>默认值</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>data_root</code></td>
<td>数据集的根目录，数据组织方式参考<a href="#2.3-构建特征库的数据组织方式">2.3节 构建特征库的数据组织方式</a></td>
<td><code>str</code></td>
<td>无</td>
</tr>
<tr>
<td><code>index_dir</code></td>
<td>特征库的保存路径。成功调用<code>build_index</code>方法后会在改路径下生成两个文件： <code>"id_map.pkl"</code> 保存了图像ID与图像特征标签之间的映射关系；<code>“vector.index”</code>存储了每张图像的特征向量</td>
<td><code>str</code></td>
<td>无</td>
</tr>
</tbody>
</table>
（3）调用通用图像识别产线对象的 `predict` 方法进行推理预测：`predict` 方法参数为 `input`，用于输入待预测数据，支持多种输入方式，具体示例如下：

<table>
<thead>
<tr>
<th>参数类型</th>
<th>参数说明</th>
</tr>
</thead>
<tbody>
<tr>
<td>Python Var</td>
<td>支持直接传入Python变量，如<code>numpy.ndarray</code>表示的图像数据。</td>
</tr>
<tr>
<td>str</td>
<td>支持传入待预测数据文件路径，如图像文件的本地路径：<code>/root/data/img.jpg</code>。</td>
</tr>
<tr>
<td>str</td>
<td>支持传入待预测数据文件URL，如图像文件的网络URL：<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/yuanqisenlin.jpeg">示例</a>。</td>
</tr>
<tr>
<td>str</td>
<td>支持传入本地目录，该目录下需包含待预测数据文件，如本地路径：<code>/root/data/</code>。</td>
</tr>
<tr>
<td>dict</td>
<td>支持传入字典类型，字典的key需与具体任务对应，如图像分类任务对应\"img\"，字典的val支持上述类型数据，例如：<code>{\"img\": \"/root/data1\"}</code>。</td>
</tr>
<tr>
<td>list</td>
<td>支持传入列表，列表元素需为上述类型数据，如<code>[numpy.ndarray, numpy.ndarray]，[\"/root/data/img1.jpg\", \"/root/data/img2.jpg\"]</code>，<code>[\"/root/data1\", \"/root/data2\"]</code>，<code>[{\"img\": \"/root/data1\"}, {\"img\": \"/root/data2/img.jpg\"}]</code>。</td>
</tr>
</tbody>
</table>
另外，`predict`方法支持参数`index_dir`用于设置检索库：
<table>
<thead>
<tr>
<th>参数类型</th>
<th>参数说明</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>index_dir</code></td>
<td>产线推理预测所用的检索库文件所在的目录，如不传入该参数，则默认使用在<code>create_pipeline()</code>中通过参数<code>index_dir</code>指定的检索库。</td>
</tr>
</tbody>
</table>
（4）调用 `predict` 方法获取预测结果：`predict` 方法为 `generator`，因此需要通过调用获得预测结果，`predict` 将方法以 batch 为单位对数据进行预测。

（5）对预测结果进行处理：每个样本的预测结果均为 `dict` 类型，且支持打印，或保存为文件，支持保存的类型与具体产线相关，如：

<table>
<thead>
<tr>
<th>方法</th>
<th>说明</th>
<th>方法参数</th>
</tr>
</thead>
<tbody>
<tr>
<td>print</td>
<td>打印结果到终端</td>
<td><code>- format_json</code>：bool类型，是否对输出内容进行使用json缩进格式化，默认为True；<br/><code>- indent</code>：int类型，json格式化设置，仅当format_json为True时有效，默认为4；<br/><code>- ensure_ascii</code>：bool类型，json格式化设置，仅当format_json为True时有效，默认为False；</td>
</tr>
<tr>
<td>save_to_json</td>
<td>将结果保存为json格式的文件</td>
<td><code>- save_path</code>：str类型，保存的文件路径，当为目录时，保存文件命名与输入文件类型命名一致；<br/><code>- indent</code>：int类型，json格式化设置，默认为4；<br/><code>- ensure_ascii</code>：bool类型，json格式化设置，默认为False；</td>
</tr>
<tr>
<td>save_to_img</td>
<td>将结果保存为图像格式的文件</td>
<td><code>- save_path</code>：str类型，保存的文件路径，当为目录时，保存文件命名与输入文件类型命名一致；</td>
</tr>
</tbody>
</table>
若您获取了配置文件，即可对通用图像识别产线各项配置进行自定义，只需要修改 `create_pipeline` 方法中的 `pipeline` 参数值为产线配置文件路径即可。

例如，若您的配置文件保存在 `./my_path/PP-ShiTuV2.yaml` ，则只需执行：

```python
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="./my_path/PP-ShiTuV2.yaml", index_dir="index_dir")

output = pipeline.predict("./drink_dataset_v2.0/test_images/")
for res in output:
    res.print()
    res.save_to_img("./output/")
```


#### 2.2.3 特征库的添加和删除操作

若您希望将更多的图像添加到特征库中，则可以调用 `append_index` 方法；删除图像特征，则可以调用 `remove_index` 方法。

```python
from paddlex import create_pipeline

pipeline = create_pipeline("PP-ShiTuV2")
pipeline.build_index(data_root="drink_dataset_v2.0/gallery/", index_dir="index_dir", index_type="IVF")
pipeline.append_index(data_root="drink_dataset_v2.0/gallery/", index_dir="index_dir", index_type="IVF")
pipeline.remove_index(data_root="drink_dataset_v2.0/gallery/", index_dir="index_dir", index_type="IVF")
```

上述方法参数说明如下：
<table>
<thead>
<tr>
<th>参数</th>
<th>参数说明</th>
<th>参数类型</th>
<th>默认值</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>data_root</code></td>
<td>要添加的数据集的根目录。数据组织方式与构建特征库时相同，参考<a href="#2.3-构建特征库的数据组织方式">2.3节 构建特征库的数据组织方式</a></td>
<td><code>str</code></td>
<td>无</td>
</tr>
<tr>
<td><code>index_dir</code></td>
<td>特征库的存储目录，在 <code>append_index</code> 和 <code>remove_index</code> 中，同时也是被修改（或删除）的特征库的路径，</td>
<td><code>str</code></td>
<td>无</td>
</tr>
<tr>
<td><code>index_type</code></td>
<td>支持 <code>HNSW32</code>、<code>IVF</code>、<code>Flat</code>。其中，<code>HNSW32</code> 检索速度较快且精度较高，但不支持 <code>remove_index()</code> 操作；<code>IVF</code> 检索速度较快但精度相对较低，支持 <code>append_index()</code> 和 <code>remove_index()</code> 操作；<code>Flat</code> 检索速度较低精度较高，支持 <code>append_index()</code> 和 <code>remove_index()</code> 操作。</td>
<td><code>str</code></td>
<td><code>HNSW32</code></td>
</tr>
<tr>
<td><code>metric_type</code></td>
<td>支持：<code>IP</code>，内积（Inner Product）；<code>L2</code>，欧几里得距离（Euclidean Distance）。</td>
<td><code>str</code></td>
<td><code>IP</code></td>
</tr>
</tbody>
</table>
### 2.3 构建特征库的数据组织方式

PaddleX 的通用图像识别产线示例需要使用预先构建好的特征库进行特征检索。如果您希望用私有数据构建特征向量库，则需要按照如下方式组织数据：

```bash
data_root             # 数据集根目录，目录名称可以改变
├── images            # 图像的保存目录，目录名称可以改变
│   │   ...
└── gallery.txt       # 特征库数据集标注文件，文件名称不可改变。每行给出待检索图像路径和图像标签，使用空格分隔，内容举例： “0/0.jpg 脉动”
```

## 3. 开发集成/部署

如果通用图像识别产线可以达到您对产线推理速度和精度的要求，您可以直接进行开发集成/部署。

若您需要将通用图像识别产线直接应用在您的Python项目中，可以参考 [2.2.2 Python脚本方式](#222-python脚本方式集成)中的示例代码。

此外，PaddleX 也提供了其他三种部署方式，详细说明如下：

🚀 <b>高性能推理</b>：在实际生产环境中，许多应用对部署策略的性能指标（尤其是响应速度）有着较严苛的标准，以确保系统的高效运行与用户体验的流畅性。为此，PaddleX 提供高性能推理插件，旨在对模型推理及前后处理进行深度性能优化，实现端到端流程的显著提速，详细的高性能推理流程请参考[PaddleX高性能推理指南](../../../pipeline_deploy/high_performance_inference.md)。

☁️ <b>服务化部署</b>：服务化部署是实际生产环境中常见的一种部署形式。通过将推理功能封装为服务，客户端可以通过网络请求来访问这些服务，以获取推理结果。PaddleX 支持用户以低成本实现产线的服务化部署，详细的服务化部署流程请参考[PaddleX服务化部署指南](../../../pipeline_deploy/service_deploy.md)。

下面是API参考和多语言服务调用示例：

<details><summary>API参考</summary>

<p>对于服务提供的所有操作：</p>
<ul>
<li>响应体以及POST请求的请求体均为JSON数据（JSON对象）。</li>
<li>当请求处理成功时，响应状态码为<code>200</code>，响应体的属性如下：</li>
</ul>
<table>
<thead>
<tr>
<th>名称</th>
<th>类型</th>
<th>含义</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>errorCode</code></td>
<td><code>integer</code></td>
<td>错误码。固定为<code>0</code>。</td>
</tr>
<tr>
<td><code>errorMsg</code></td>
<td><code>string</code></td>
<td>错误说明。固定为<code>"Success"</code>。</td>
</tr>
</tbody>
</table>
<p>响应体还可能有<code>result</code>属性，类型为<code>object</code>，其中存储操作结果信息。</p>
<ul>
<li>当请求处理未成功时，响应体的属性如下：</li>
</ul>
<table>
<thead>
<tr>
<th>名称</th>
<th>类型</th>
<th>含义</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>errorCode</code></td>
<td><code>integer</code></td>
<td>错误码。与响应状态码相同。</td>
</tr>
<tr>
<td><code>errorMsg</code></td>
<td><code>string</code></td>
<td>错误说明。</td>
</tr>
</tbody>
</table>
<p>服务提供的操作如下：</p>
<ul>
<li><b><code>infer</code></b></li>
</ul>
<p>获取图像OCR结果。</p>
<p><code>POST /ocr</code></p>
<ul>
<li>请求体的属性如下：</li>
</ul>
<table>
<thead>
<tr>
<th>名称</th>
<th>类型</th>
<th>含义</th>
<th>是否必填</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>image</code></td>
<td><code>string</code></td>
<td>服务可访问的图像文件的URL或图像文件内容的Base64编码结果。</td>
<td>是</td>
</tr>
<tr>
<td><code>inferenceParams</code></td>
<td><code>object</code></td>
<td>推理参数。</td>
<td>否</td>
</tr>
</tbody>
</table>
<p><code>inferenceParams</code>的属性如下：</p>
<table>
<thead>
<tr>
<th>名称</th>
<th>类型</th>
<th>含义</th>
<th>是否必填</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>maxLongSide</code></td>
<td><code>integer</code></td>
<td>推理时，若文本检测模型的输入图像较长边的长度大于<code>maxLongSide</code>，则将对图像进行缩放，使其较长边的长度等于<code>maxLongSide</code>。</td>
<td>否</td>
</tr>
</tbody>
</table>
<ul>
<li>请求处理成功时，响应体的<code>result</code>具有如下属性：</li>
</ul>
<table>
<thead>
<tr>
<th>名称</th>
<th>类型</th>
<th>含义</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>texts</code></td>
<td><code>array</code></td>
<td>文本位置、内容和得分。</td>
</tr>
<tr>
<td><code>image</code></td>
<td><code>string</code></td>
<td>OCR结果图，其中标注检测到的文本位置。图像为JPEG格式，使用Base64编码。</td>
</tr>
</tbody>
</table>
<p><code>texts</code>中的每个元素为一个<code>object</code>，具有如下属性：</p>
<table>
<thead>
<tr>
<th>名称</th>
<th>类型</th>
<th>含义</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>poly</code></td>
<td><code>array</code></td>
<td>文本位置。数组中元素依次为包围文本的多边形的顶点坐标。</td>
</tr>
<tr>
<td><code>text</code></td>
<td><code>string</code></td>
<td>文本内容。</td>
</tr>
<tr>
<td><code>score</code></td>
<td><code>number</code></td>
<td>文本识别得分。</td>
</tr>
</tbody>
</table>
<p><code>result</code>示例如下：</p>
<pre><code class="language-json">{
&quot;texts&quot;: [
{
&quot;poly&quot;: [
[
444,
244
],
[
705,
244
],
[
705,
311
],
[
444,
311
]
],
&quot;text&quot;: &quot;北京南站&quot;,
&quot;score&quot;: 0.9
},
{
&quot;poly&quot;: [
[
992,
248
],
[
1263,
251
],
[
1263,
318
],
[
992,
315
]
],
&quot;text&quot;: &quot;天津站&quot;,
&quot;score&quot;: 0.5
}
],
&quot;image&quot;: &quot;xxxxxx&quot;
}
</code></pre></details>

<details><summary>多语言调用服务示例</summary>

<details>
<summary>Python</summary>


<pre><code class="language-python">import base64
import requests

API_URL = &quot;http://localhost:8080/ocr&quot; # 服务URL
image_path = &quot;./demo.jpg&quot;
output_image_path = &quot;./out.jpg&quot;

# 对本地图像进行Base64编码
with open(image_path, &quot;rb&quot;) as file:
    image_bytes = file.read()
    image_data = base64.b64encode(image_bytes).decode(&quot;ascii&quot;)

payload = {&quot;image&quot;: image_data}  # Base64编码的文件内容或者图像URL

# 调用API
response = requests.post(API_URL, json=payload)

# 处理接口返回数据
assert response.status_code == 200
result = response.json()[&quot;result&quot;]
with open(output_image_path, &quot;wb&quot;) as file:
    file.write(base64.b64decode(result[&quot;image&quot;]))
print(f&quot;Output image saved at {output_image_path}&quot;)
print(&quot;\nDetected texts:&quot;)
print(result[&quot;texts&quot;])
</code></pre></details>

<details><summary>C++</summary>

<pre><code class="language-cpp">#include &lt;iostream&gt;
#include &quot;cpp-httplib/httplib.h&quot; // https://github.com/Huiyicc/cpp-httplib
#include &quot;nlohmann/json.hpp&quot; // https://github.com/nlohmann/json
#include &quot;base64.hpp&quot; // https://github.com/tobiaslocker/base64

int main() {
    httplib::Client client(&quot;localhost:8080&quot;);
    const std::string imagePath = &quot;./demo.jpg&quot;;
    const std::string outputImagePath = &quot;./out.jpg&quot;;

    httplib::Headers headers = {
        {&quot;Content-Type&quot;, &quot;application/json&quot;}
    };

    // 对本地图像进行Base64编码
    std::ifstream file(imagePath, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector&lt;char&gt; buffer(size);
    if (!file.read(buffer.data(), size)) {
        std::cerr &lt;&lt; &quot;Error reading file.&quot; &lt;&lt; std::endl;
        return 1;
    }
    std::string bufferStr(reinterpret_cast&lt;const char*&gt;(buffer.data()), buffer.size());
    std::string encodedImage = base64::to_base64(bufferStr);

    nlohmann::json jsonObj;
    jsonObj[&quot;image&quot;] = encodedImage;
    std::string body = jsonObj.dump();

    // 调用API
    auto response = client.Post(&quot;/ocr&quot;, headers, body, &quot;application/json&quot;);
    // 处理接口返回数据
    if (response &amp;&amp; response-&gt;status == 200) {
        nlohmann::json jsonResponse = nlohmann::json::parse(response-&gt;body);
        auto result = jsonResponse[&quot;result&quot;];

        encodedImage = result[&quot;image&quot;];
        std::string decodedString = base64::from_base64(encodedImage);
        std::vector&lt;unsigned char&gt; decodedImage(decodedString.begin(), decodedString.end());
        std::ofstream outputImage(outPutImagePath, std::ios::binary | std::ios::out);
        if (outputImage.is_open()) {
            outputImage.write(reinterpret_cast&lt;char*&gt;(decodedImage.data()), decodedImage.size());
            outputImage.close();
            std::cout &lt;&lt; &quot;Output image saved at &quot; &lt;&lt; outPutImagePath &lt;&lt; std::endl;
        } else {
            std::cerr &lt;&lt; &quot;Unable to open file for writing: &quot; &lt;&lt; outPutImagePath &lt;&lt; std::endl;
        }

        auto texts = result[&quot;texts&quot;];
        std::cout &lt;&lt; &quot;\nDetected texts:&quot; &lt;&lt; std::endl;
        for (const auto&amp; text : texts) {
            std::cout &lt;&lt; text &lt;&lt; std::endl;
        }
    } else {
        std::cout &lt;&lt; &quot;Failed to send HTTP request.&quot; &lt;&lt; std::endl;
        return 1;
    }

    return 0;
}
</code></pre></details>

<details><summary>Java</summary>

<pre><code class="language-java">import okhttp3.*;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Base64;

public class Main {
    public static void main(String[] args) throws IOException {
        String API_URL = &quot;http://localhost:8080/ocr&quot;; // 服务URL
        String imagePath = &quot;./demo.jpg&quot;; // 本地图像
        String outputImagePath = &quot;./out.jpg&quot;; // 输出图像

        // 对本地图像进行Base64编码
        File file = new File(imagePath);
        byte[] fileContent = java.nio.file.Files.readAllBytes(file.toPath());
        String imageData = Base64.getEncoder().encodeToString(fileContent);

        ObjectMapper objectMapper = new ObjectMapper();
        ObjectNode params = objectMapper.createObjectNode();
        params.put(&quot;image&quot;, imageData); // Base64编码的文件内容或者图像URL

        // 创建 OkHttpClient 实例
        OkHttpClient client = new OkHttpClient();
        MediaType JSON = MediaType.Companion.get(&quot;application/json; charset=utf-8&quot;);
        RequestBody body = RequestBody.Companion.create(params.toString(), JSON);
        Request request = new Request.Builder()
                .url(API_URL)
                .post(body)
                .build();

        // 调用API并处理接口返回数据
        try (Response response = client.newCall(request).execute()) {
            if (response.isSuccessful()) {
                String responseBody = response.body().string();
                JsonNode resultNode = objectMapper.readTree(responseBody);
                JsonNode result = resultNode.get(&quot;result&quot;);
                String base64Image = result.get(&quot;image&quot;).asText();
                JsonNode texts = result.get(&quot;texts&quot;);

                byte[] imageBytes = Base64.getDecoder().decode(base64Image);
                try (FileOutputStream fos = new FileOutputStream(outputImagePath)) {
                    fos.write(imageBytes);
                }
                System.out.println(&quot;Output image saved at &quot; + outputImagePath);
                System.out.println(&quot;\nDetected texts: &quot; + texts.toString());
            } else {
                System.err.println(&quot;Request failed with code: &quot; + response.code());
            }
        }
    }
}
</code></pre></details>

<details><summary>Go</summary>

<pre><code class="language-go">package main

import (
    &quot;bytes&quot;
    &quot;encoding/base64&quot;
    &quot;encoding/json&quot;
    &quot;fmt&quot;
    &quot;io/ioutil&quot;
    &quot;net/http&quot;
)

func main() {
    API_URL := &quot;http://localhost:8080/ocr&quot;
    imagePath := &quot;./demo.jpg&quot;
    outputImagePath := &quot;./out.jpg&quot;

    // 对本地图像进行Base64编码
    imageBytes, err := ioutil.ReadFile(imagePath)
    if err != nil {
        fmt.Println(&quot;Error reading image file:&quot;, err)
        return
    }
    imageData := base64.StdEncoding.EncodeToString(imageBytes)

    payload := map[string]string{&quot;image&quot;: imageData} // Base64编码的文件内容或者图像URL
    payloadBytes, err := json.Marshal(payload)
    if err != nil {
        fmt.Println(&quot;Error marshaling payload:&quot;, err)
        return
    }

    // 调用API
    client := &amp;http.Client{}
    req, err := http.NewRequest(&quot;POST&quot;, API_URL, bytes.NewBuffer(payloadBytes))
    if err != nil {
        fmt.Println(&quot;Error creating request:&quot;, err)
        return
    }

    res, err := client.Do(req)
    if err != nil {
        fmt.Println(&quot;Error sending request:&quot;, err)
        return
    }
    defer res.Body.Close()

    // 处理接口返回数据
    body, err := ioutil.ReadAll(res.Body)
    if err != nil {
        fmt.Println(&quot;Error reading response body:&quot;, err)
        return
    }
    type Response struct {
        Result struct {
            Image      string   `json:&quot;image&quot;`
            Texts []map[string]interface{} `json:&quot;texts&quot;`
        } `json:&quot;result&quot;`
    }
    var respData Response
    err = json.Unmarshal([]byte(string(body)), &amp;respData)
    if err != nil {
        fmt.Println(&quot;Error unmarshaling response body:&quot;, err)
        return
    }

    outputImageData, err := base64.StdEncoding.DecodeString(respData.Result.Image)
    if err != nil {
        fmt.Println(&quot;Error decoding base64 image data:&quot;, err)
        return
    }
    err = ioutil.WriteFile(outputImagePath, outputImageData, 0644)
    if err != nil {
        fmt.Println(&quot;Error writing image to file:&quot;, err)
        return
    }
    fmt.Printf(&quot;Image saved at %s.jpg\n&quot;, outputImagePath)
    fmt.Println(&quot;\nDetected texts:&quot;)
    for _, text := range respData.Result.Texts {
        fmt.Println(text)
    }
}
</code></pre></details>

<details><summary>C#</summary>

<pre><code class="language-csharp">using System;
using System.IO;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json.Linq;

class Program
{
    static readonly string API_URL = &quot;http://localhost:8080/ocr&quot;;
    static readonly string imagePath = &quot;./demo.jpg&quot;;
    static readonly string outputImagePath = &quot;./out.jpg&quot;;

    static async Task Main(string[] args)
    {
        var httpClient = new HttpClient();

        // 对本地图像进行Base64编码
        byte[] imageBytes = File.ReadAllBytes(imagePath);
        string image_data = Convert.ToBase64String(imageBytes);

        var payload = new JObject{ { &quot;image&quot;, image_data } }; // Base64编码的文件内容或者图像URL
        var content = new StringContent(payload.ToString(), Encoding.UTF8, &quot;application/json&quot;);

        // 调用API
        HttpResponseMessage response = await httpClient.PostAsync(API_URL, content);
        response.EnsureSuccessStatusCode();

        // 处理接口返回数据
        string responseBody = await response.Content.ReadAsStringAsync();
        JObject jsonResponse = JObject.Parse(responseBody);

        string base64Image = jsonResponse[&quot;result&quot;][&quot;image&quot;].ToString();
        byte[] outputImageBytes = Convert.FromBase64String(base64Image);

        File.WriteAllBytes(outputImagePath, outputImageBytes);
        Console.WriteLine($&quot;Output image saved at {outputImagePath}&quot;);
        Console.WriteLine(&quot;\nDetected texts:&quot;);
        Console.WriteLine(jsonResponse[&quot;result&quot;][&quot;texts&quot;].ToString());
    }
}
</code></pre></details>

<details><summary>Node.js</summary>

<pre><code class="language-js">const axios = require('axios');
const fs = require('fs');

const API_URL = 'http://localhost:8080/ocr'
const imagePath = './demo.jpg'
const outputImagePath = &quot;./out.jpg&quot;;

let config = {
   method: 'POST',
   maxBodyLength: Infinity,
   url: API_URL,
   data: JSON.stringify({
    'image': encodeImageToBase64(imagePath)  // Base64编码的文件内容或者图像URL
  })
};

// 对本地图像进行Base64编码
function encodeImageToBase64(filePath) {
  const bitmap = fs.readFileSync(filePath);
  return Buffer.from(bitmap).toString('base64');
}

// 调用API
axios.request(config)
.then((response) =&gt; {
    // 处理接口返回数据
    const result = response.data[&quot;result&quot;];
    const imageBuffer = Buffer.from(result[&quot;image&quot;], 'base64');
    fs.writeFile(outputImagePath, imageBuffer, (err) =&gt; {
      if (err) throw err;
      console.log(`Output image saved at ${outputImagePath}`);
    });
    console.log(&quot;\nDetected texts:&quot;);
    console.log(result[&quot;texts&quot;]);
})
.catch((error) =&gt; {
  console.log(error);
});
</code></pre></details>

<details><summary>PHP</summary>

<pre><code class="language-php">&lt;?php

$API_URL = &quot;http://localhost:8080/ocr&quot;; // 服务URL
$image_path = &quot;./demo.jpg&quot;;
$output_image_path = &quot;./out.jpg&quot;;

// 对本地图像进行Base64编码
$image_data = base64_encode(file_get_contents($image_path));
$payload = array(&quot;image&quot; =&gt; $image_data); // Base64编码的文件内容或者图像URL

// 调用API
$ch = curl_init($API_URL);
curl_setopt($ch, CURLOPT_POST, true);
curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($payload));
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
$response = curl_exec($ch);
curl_close($ch);

// 处理接口返回数据
$result = json_decode($response, true)[&quot;result&quot;];
file_put_contents($output_image_path, base64_decode($result[&quot;image&quot;]));
echo &quot;Output image saved at &quot; . $output_image_path . &quot;\n&quot;;
echo &quot;\nDetected texts:\n&quot;;
print_r($result[&quot;texts&quot;]);

?&gt;
</code></pre></details>
</details>
<br/>

📱 <b>端侧部署</b>：端侧部署是一种将计算和数据处理功能放在用户设备本身上的方式，设备可以直接处理数据，而不需要依赖远程的服务器。PaddleX 支持将模型部署在 Android 等端侧设备上，详细的端侧部署流程请参考[PaddleX端侧部署指南](../../../pipeline_deploy/edge_deploy.md)。
您可以根据需要选择合适的方式部署模型产线，进而进行后续的 AI 应用集成。


## 4. 二次开发

如果通用图像识别产线提供的默认模型权重在您的场景中，精度或速度不满意，您可以尝试利用<b>您自己拥有的特定领域或应用场景的数据</b>对现有模型进行进一步的<b>微调</b>，以提升通用该产线的在您的场景中的识别效果。

### 4.1 模型微调

由于通用图像识别产线包含两个模块（主体检测模块和图像特征模块），模型产线的效果不及预期可能来自于其中任何一个模块。

您可以对识别效果差的图片进行分析，如果在分析过程中发现有较多的主体目标未被检测出来，那么可能是主体检测模型存在不足，您需要参考[主体检测模块开发教程](../../../module_usage/tutorials/cv_modules/object_detection.md)中的[二次开发](../../../module_usage/tutorials/cv_modules/object_detection.md#四二次开发)章节，使用您的私有数据集对主体检测模型进行微调；如果在已检测到的主体出现匹配错误，这表明图像特征模型需要进一步改进，您需要参考[图像特征模块开发教程](../../../module_usage/tutorials/cv_modules/image_feature.md)中的[二次开发](../../../module_usage/tutorials/cv_modules/image_feature.md#四二次开发)章节,对图像特征模型进行微调。

### 4.2 模型应用

当您使用私有数据集完成微调训练后，可获得本地模型权重文件。

若您需要使用微调后的模型权重，只需对产线配置文件做修改，将微调后模型权重的本地路径替换至产线配置文件中的对应位置即可：

```yaml
Pipeline:
  device: "gpu:0"
  det_model: "./PP-ShiTuV2_det_infer/"        #可修改为微调后主体检测模型的本地路径
  rec_model: "./PP-ShiTuV2_rec_infer/"        #可修改为微调后图像特征模型的本地路径
  det_batch_size: 1
  rec_batch_size: 1
  device: gpu
```
随后， 参考[2.2 本地体验](#22-本地体验)中的命令行方式或Python脚本方式，加载修改后的产线配置文件即可。

##  5. 多硬件支持

PaddleX 支持英伟达 GPU、昆仑芯 XPU、昇腾 NPU和寒武纪 MLU 等多种主流硬件设备，<b>仅需修改 `--device`参数</b>即可完成不同硬件之间的无缝切换。

例如，使用Python运行通用图像识别产线时，将运行设备从英伟达 GPU 更改为昇腾 NPU，仅需将脚本中的 `device` 修改为 npu 即可：

```python
from paddlex import create_pipeline

pipeline = create_pipeline(
    pipeline="PP-ShiTuV2",
    device="npu:0" # gpu:0 --> npu:0
    )
```

若您想在更多种类的硬件上使用通用图像识别产线，请参考[PaddleX多硬件使用指南](../../../other_devices_support/multi_devices_use_guide.md)。
