---
comments: true
---

# PaddleX Model List (Hygon DCU)

PaddleX incorporates multiple pipelines, each containing several modules, and each module encompasses various models. The specific models to use can be selected based on the benchmark data below. If you prioritize model accuracy, choose models with higher accuracy. If you prioritize model storage size, select models with smaller storage sizes.

## Image Classification Module
<table>
<thead>
<tr>
<th>Model Name</th>
<th>Top-1 Accuracy (%)</th>
<th>Model Storage Size (M)</th>
</tr>
</thead>
<tbody>
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
<b>Note: The above accuracy metrics are Top-1 Accuracy on the [ImageNet-1k](https://www.image-net.org/index.php) validation set.</b>

## Semantic Segmentation Module
<table>
<thead>
<tr>
<th>Model Name</th>
<th>mIoU (%)</th>
<th>Model Storage Size (M)</th>
</tr>
</thead>
<tbody>
<tr>
<td>Deeplabv3_Plus-R50</td>
<td>80.36</td>
<td>94.9 M</td>
</tr>
<tr>
<td>Deeplabv3_Plus-R101</td>
<td>81.10</td>
<td>162.5 M</td>
</tr>
</tbody>
</table>
<b>Note: The above accuracy metrics are mIoU on the [Cityscapes](https://www.cityscapes-dataset.com/) dataset.</b>
