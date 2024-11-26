---
comments: true
---

# PaddleX模型列表（燧原 GCU）

PaddleX 内置了多条产线，每条产线都包含了若干模块，每个模块包含若干模型，具体使用哪些模型，您可以根据下边的 benchmark 数据来选择。如您更考虑模型精度，请选择精度较高的模型，如您更考虑模型存储大小，请选择存储大小较小的模型。

## 图像分类模块
<table>
<thead>
<tr>
<th>模型名称</th>
<th>Top1 Acc（%）</th>
<th>模型存储大小（M)</th>
<th>模型下载链接</th></tr>
</thead>
<tbody>
<tr>
<td>ResNet50</td>
<td>76.96</td>
<td>90.8 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/ResNet50_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ResNet50_pretrained.pdparams">训练模型</a></td></tr>
</tbody>
</table>
<b>注：以上精度指标为</b>[ImageNet-1k](https://www.image-net.org/index.php)<b>验证集 Top1 Acc。</b>
