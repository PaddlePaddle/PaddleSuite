---
comments: true
---

# PaddleX模型列表（昆仑 XPU）

PaddleX 内置了多条产线，每条产线都包含了若干模块，每个模块包含若干模型，具体使用哪些模型，您可以根据下边的 benchmark 数据来选择。如您更考虑模型精度，请选择精度较高的模型，如您更考虑模型存储大小，请选择存储大小较小的模型。

## 图像分类模块
<table>
<thead>
<tr>
<th>模型名称</th>
<th>Top1 Acc（%）</th>
<th>模型存储大小（M)</th>
</tr>
</thead>
<tbody>
<tr>
<td>MobileNetV3_large_x0_5</td>
<td>69.2</td>
<td>9.6 M</td>
</tr>
<tr>
<td>MobileNetV3_large_x0_35</td>
<td>64.3</td>
<td>7.5 M</td>
</tr>
<tr>
<td>MobileNetV3_large_x0_75</td>
<td>73.1</td>
<td>14.0 M</td>
</tr>
<tr>
<td>MobileNetV3_large_x1_0</td>
<td>75.3</td>
<td>19.5 M</td>
</tr>
<tr>
<td>MobileNetV3_large_x1_25</td>
<td>76.4</td>
<td>26.5 M</td>
</tr>
<tr>
<td>MobileNetV3_small_x0_5</td>
<td>59.2</td>
<td>6.8 M</td>
</tr>
<tr>
<td>MobileNetV3_small_x0_35</td>
<td>53.0</td>
<td>6.0 M</td>
</tr>
<tr>
<td>MobileNetV3_small_x0_75</td>
<td>66.0</td>
<td>8.5 M</td>
</tr>
<tr>
<td>MobileNetV3_small_x1_0</td>
<td>68.2</td>
<td>10.5 M</td>
</tr>
<tr>
<td>MobileNetV3_small_x1_25</td>
<td>70.7</td>
<td>13.0 M</td>
</tr>
<tr>
<td>PP-HGNet_small</td>
<td>81.51</td>
<td>86.5 M</td>
</tr>
<tr>
<td>PP-LCNet_x0_5</td>
<td>63.14</td>
<td>6.7 M</td>
</tr>
<tr>
<td>PP-LCNet_x0_25</td>
<td>51.86</td>
<td>5.5 M</td>
</tr>
<tr>
<td>PP-LCNet_x0_35</td>
<td>58.09</td>
<td>5.9 M</td>
</tr>
<tr>
<td>PP-LCNet_x0_75</td>
<td>68.18</td>
<td>8.4 M</td>
</tr>
<tr>
<td>PP-LCNet_x1_0</td>
<td>71.32</td>
<td>10.5 M</td>
</tr>
<tr>
<td>PP-LCNet_x1_5</td>
<td>73.71</td>
<td>16.0 M</td>
</tr>
<tr>
<td>PP-LCNet_x2_0</td>
<td>75.18</td>
<td>23.2 M</td>
</tr>
<tr>
<td>PP-LCNet_x2_5</td>
<td>76.60</td>
<td>32.1 M</td>
</tr>
<tr>
<td>ResNet18</td>
<td>71.0</td>
<td>41.5 M</td>
</tr>
<tr>
<td>ResNet34</td>
<td>74.6</td>
<td>77.3 M</td>
</tr>
<tr>
<td>ResNet50</td>
<td>76.5</td>
<td>90.8 M</td>
</tr>
<tr>
<td>ResNet101</td>
<td>77.6</td>
<td>158.7 M</td>
</tr>
<tr>
<td>ResNet152</td>
<td>78.3</td>
<td>214.2 M</td>
</tr>
</tbody>
</table>
<b>注：以上精度指标为</b>[ImageNet-1k](https://www.image-net.org/index.php)<b>验证集 Top1 Acc。</b>

## 目标检测模块
<table>
<thead>
<tr>
<th>模型名称</th>
<th>mAP（%）</th>
<th>模型存储大小（M)</th>
</tr>
</thead>
<tbody>
<tr>
<td>PicoDet-L</td>
<td>42.6</td>
<td>20.9 M</td>
</tr>
<tr>
<td>PicoDet-S</td>
<td>29.1</td>
<td>4.4 M</td>
</tr>
<tr>
<td>PP-YOLOE_plus-L</td>
<td>52.9</td>
<td>185.3 M</td>
</tr>
<tr>
<td>PP-YOLOE_plus-M</td>
<td>49.8</td>
<td>83.2 M</td>
</tr>
<tr>
<td>PP-YOLOE_plus-S</td>
<td>43.7</td>
<td>28.3 M</td>
</tr>
<tr>
<td>PP-YOLOE_plus-X</td>
<td>54.7</td>
<td>349.4 M</td>
</tr>
</tbody>
</table>
<b>注：以上精度指标为</b>[COCO2017](https://cocodataset.org/#home)<b>验证集 mAP(0.5:0.95)。</b>

## 语义分割模块
<table>
<thead>
<tr>
<th>模型名称</th>
<th>mloU（%）</th>
<th>模型存储大小（M)</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-LiteSeg-T</td>
<td>73.10</td>
<td>28.5 M</td>
</tr>
</tbody>
</table>
<b>注：以上精度指标为</b>[Cityscapes](https://www.cityscapes-dataset.com/)<b>数据集 mloU。</b>

## 文本检测模块
<table>
<thead>
<tr>
<th>模型名称</th>
<th>检测Hmean（%）</th>
<th>模型存储大小（M)</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-OCRv4_mobile_det</td>
<td>77.79</td>
<td>4.2 M</td>
</tr>
<tr>
<td>PP-OCRv4_server_det ）</td>
<td>82.69</td>
<td>100.1M</td>
</tr>
</tbody>
</table>
<b>注：以上精度指标的评估集是 PaddleOCR 自建的中文数据集，覆盖街景、网图、文档、手写多个场景，其中检测包含 500 张图片。</b>

## 文本识别模块
<table>
<thead>
<tr>
<th>模型名称</th>
<th>识别Avg Accuracy(%)</th>
<th>模型存储大小（M)</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-OCRv4_mobile_rec</td>
<td>78.20</td>
<td>10.6 M</td>
</tr>
<tr>
<td>PP-OCRv4_server_rec</td>
<td>79.20</td>
<td>71.2 M</td>
</tr>
</tbody>
</table>
<b>注：以上精度指标的评估集是 PaddleOCR 自建的中文数据集，覆盖街景、网图、文档、手写多个场景，其中文本识别包含 1.1w 张图片。</b>

## 版面区域分析模块
<table>
<thead>
<tr>
<th>模型名称</th>
<th>mAP（%）</th>
<th>模型存储大小（M)</th>
</tr>
</thead>
<tbody>
<tr>
<td>PicoDet_layout_1x</td>
<td>86.8</td>
<td>7.4M</td>
</tr>
</tbody>
</table>
<b>注：以上精度指标的评估集是 PaddleOCR 自建的版面区域分析数据集，包含 1w 张图片。</b>

## 时序预测模块
<table>
<thead>
<tr>
<th>模型名称</th>
<th>mse</th>
<th>mae</th>
<th>模型存储大小（M)</th>
</tr>
</thead>
<tbody>
<tr>
<td>DLinear</td>
<td>0.382</td>
<td>0.394</td>
<td>72K</td>
</tr>
<tr>
<td>NLinear</td>
<td>0.386</td>
<td>0.392</td>
<td>40K</td>
</tr>
<tr>
<td>RLinear</td>
<td>0.384</td>
<td>0.392</td>
<td>40K</td>
</tr>
</tbody>
</table>
<b>注：以上精度指标测量自</b>[ETTH1](https://paddle-model-ecology.bj.bcebos.com/paddlex/data/Etth1.tar)<b>数据集 </b><b>（在测试集test.csv上的评测结果）</b><b>。</b>
