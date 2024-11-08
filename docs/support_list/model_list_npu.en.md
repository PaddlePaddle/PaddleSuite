---
comments: true
---

# PaddleX Model List (Huawei Ascend NPU)

PaddleX incorporates multiple pipelines, each containing several modules, and each module encompasses various models. You can select the appropriate models based on the benchmark data below. If you prioritize model accuracy, choose models with higher accuracy. If you prioritize model size, select models with smaller storage requirements.

## Image Classification Module
<table>
<thead>
<tr>
<th>Model Name</th>
<th>Top-1 Accuracy (%)</th>
<th>Model Size (M)</th>
</tr>
</thead>
<tbody>
<tr>
<td>CLIP_vit_base_patch16_224</td>
<td>85.36</td>
<td>306.5 M</td>
</tr>
<tr>
<td>CLIP_vit_large_patch14_224</td>
<td>88.1</td>
<td>1.04 G</td>
</tr>
<tr>
<td>ConvNeXt_base_224</td>
<td>83.84</td>
<td>313.9 M</td>
</tr>
<tr>
<td>ConvNeXt_base_384</td>
<td>84.90</td>
<td>313.9 M</td>
</tr>
<tr>
<td>ConvNeXt_large_224</td>
<td>84.26</td>
<td>700.7 M</td>
</tr>
<tr>
<td>ConvNeXt_large_384</td>
<td>85.27</td>
<td>700.7 M</td>
</tr>
<tr>
<td>ConvNeXt_small</td>
<td>83.13</td>
<td>178.0 M</td>
</tr>
<tr>
<td>ConvNeXt_tiny</td>
<td>82.03</td>
<td>101.4 M</td>
</tr>
<tr>
<td>MobileNetV1_x0_75</td>
<td>68.8</td>
<td>9.3 M</td>
</tr>
<tr>
<td>MobileNetV1_x1_0</td>
<td>71.0</td>
<td>15.2 M</td>
</tr>
<tr>
<td>MobileNetV2_x0_5</td>
<td>65.0</td>
<td>7.1 M</td>
</tr>
<tr>
<td>MobileNetV2_x0_25</td>
<td>53.2</td>
<td>5.5 M</td>
</tr>
<tr>
<td>MobileNetV2_x1_0</td>
<td>72.2</td>
<td>12.6 M</td>
</tr>
<tr>
<td>MobileNetV2_x1_5</td>
<td>74.1</td>
<td>25.0 M</td>
</tr>
<tr>
<td>MobileNetV2_x2_0</td>
<td>75.2</td>
<td>41.2 M</td>
</tr>
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
<td>PP-HGNet_base</td>
<td>85.0</td>
<td>249.4 M</td>
</tr>
<tr>
<td>PP-HGNet_small</td>
<td>81.51</td>
<td>86.5 M</td>
</tr>
<tr>
<td>PP-HGNet_tiny</td>
<td>79.83</td>
<td>52.4 M</td>
</tr>
<tr>
<td>PP-HGNetV2-B0</td>
<td>77.77</td>
<td>21.4 M</td>
</tr>
<tr>
<td>PP-HGNetV2-B1</td>
<td>79.18</td>
<td>22.6 M</td>
</tr>
<tr>
<td>PP-HGNetV2-B2</td>
<td>81.74</td>
<td>39.9 M</td>
</tr>
<tr>
<td>PP-HGNetV2-B3</td>
<td>82.98</td>
<td>57.9 M</td>
</tr>
<tr>
<td>PP-HGNetV2-B4</td>
<td>83.57</td>
<td>70.4 M</td>
</tr>
<tr>
<td>PP-HGNetV2-B5</td>
<td>84.75</td>
<td>140.8 M</td>
</tr>
<tr>
<td>PP-HGNetV2-B6</td>
<td>86.30</td>
<td>268.4 M</td>
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
<td>PP-LCNetV2_base</td>
<td>77.05</td>
<td>23.7 M</td>
</tr>
<tr>
<td>ResNet18_vd</td>
<td>72.3</td>
<td>41.5 M</td>
</tr>
<tr>
<td>ResNet18</td>
<td>71.0</td>
<td>41.5 M</td>
</tr>
<tr>
<td>ResNet34_vd</td>
<td>76.0</td>
<td>77.3 M</td>
</tr>
<tr>
<td>ResNet34</td>
<td>74.6</td>
<td>77.3 M</td>
</tr>
<tr>
<td>ResNet50_vd</td>
<td>79.1</td>
<td>90.8 M</td>
</tr>
<tr>
<td>ResNet50</td>
<td>76.5</td>
<td>90.8 M</td>
</tr>
<tr>
<td>ResNet101_vd</td>
<td>80.2</td>
<td>158.4 M</td>
</tr>
<tr>
<td>ResNet101</td>
<td>77.6</td>
<td>158.7 M</td>
</tr>
<tr>
<td>ResNet152_vd</td>
<td>80.6</td>
<td>214.3 M</td>
</tr>
<tr>
<td>ResNet152</td>
<td>78.3</td>
<td>214.2 M</td>
</tr>
<tr>
<td>ResNet200_vd</td>
<td>80.9</td>
<td>266.0 M</td>
</tr>
<tr>
<td>SwinTransformer_base_patch4_window7_224</td>
<td>83.37</td>
<td>310.5 M</td>
</tr>
<tr>
<td>SwinTransformer_small_patch4_window7_224</td>
<td>83.21</td>
<td>175.6 M</td>
</tr>
<tr>
<td>SwinTransformer_tiny_patch4_window7_224</td>
<td>81.10</td>
<td>100.1 M</td>
</tr>
</tbody>
</table>
<b>Note: The above accuracy metrics refer to Top-1 Accuracy on the [ImageNet-1k](https://www.image-net.org/index.php) validation set.</b>

## Object Detection Module
<table>
<thead>
<tr>
<th>Model Name</th>
<th>mAP (%)</th>
<th>Model Size (M)</th>
</tr>
</thead>
<tbody>
<tr>
<td>CenterNet-DLA-34</td>
<td>37.6</td>
<td>75.4 M</td>
</tr>
<tr>
<td>CenterNet-ResNet50</td>
<td>38.9</td>
<td>319.7 M</td>
</tr>
<tr>
<td>DETR-R50</td>
<td>42.3</td>
<td>159.3 M</td>
</tr>
<tr>
<td>FasterRCNN-ResNet34-FPN</td>
<td>37.8</td>
<td>137.5 M</td>
</tr>
<tr>
<td>FasterRCNN-ResNet50-FPN</td>
<td>38.4</td>
<td>148.1 M</td>
</tr>
<tr>
<td>FasterRCNN-ResNet50-vd-FPN</td>
<td>39.5</td>
<td>148.1 M</td>
</tr>
<tr>
<td>FasterRCNN-ResNet50-vd-SSLDv2-FPN</td>
<td>41.4</td>
<td>148.1 M</td>
</tr>
<tr>
<td>FasterRCNN-ResNet101-FPN</td>
<td>41.4</td>
<td>216.3 M</td>
</tr>
<tr>
<td>FCOS-ResNet50</td>
<td>39.6</td>
<td>124.2 M</td>
</tr>
<tr>
<td>PicoDet-L</td>
<td>42.6</td>
<td>20.9 M</td>
</tr>
<tr>
<td>PicoDet-M</td>
<td>37.5</td>
<td>16.8 M</td>
</tr>
<tr>
<td>PicoDet-S</td>
<td>29.1</td>
<td>4.4 M</td>
</tr>
<tr>
<td>PicoDet-XS</td>
<td>26.2</td>
<td>5.7M</td>
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
<tr>
<td>RT-DETR-H</td>
<td>56.3</td>
<td>435.8 M</td>
</tr>
<tr>
<td>RT-DETR-L</td>
<td>53.0</td>
<td>113.7 M</td>
</tr>
<tr>
<td>RT-DETR-R18</td>
<td>46.5</td>
<td>70.7 M</td>
</tr>
<tr>
<td>RT-DETR-R50</td>
<td>53.1</td>
<td>149.1 M</td>
</tr>
<tr>
<td>RT-DETR-X</td>
<td>54.8</td>
<td>232.9 M</td>
</tr>
<tr>
<td>YOLOv3-DarkNet53</td>
<td>39.1</td>
<td>219.7 M</td>
</tr>
<tr>
<td>YOLOv3-MobileNetV3</td>
<td>31.4</td>
<td>83.8 M</td>
</tr>
<tr>
<td>YOLOv3-ResNet50_vd_DCN</td>
<td>40.6</td>
<td>163.0 M</td>
</tr>
</tbody>
</table>
<b>Note: The above accuracy metrics are for</b> [COCO2017](https://cocodataset.org/#home) <b>validation set mAP(0.5:0.95).</b>

## Semantic Segmentation Module
<table>
<thead>
<tr>
<th>Model Name</th>
<th>mIoU (%)</th>
<th>Model Size (M)</th>
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
<tr>
<td>Deeplabv3-R50</td>
<td>79.90</td>
<td>138.3 M</td>
</tr>
<tr>
<td>Deeplabv3-R101</td>
<td>80.85</td>
<td>205.9 M</td>
</tr>
<tr>
<td>OCRNet_HRNet-W48</td>
<td>82.15</td>
<td>249.8 M</td>
</tr>
<tr>
<td>PP-LiteSeg-T</td>
<td>73.10</td>
<td>28.5 M</td>
</tr>
</tbody>
</table>
<b>Note: The above accuracy metrics are for</b> [Cityscapes](https://www.cityscapes-dataset.com/) <b>dataset mIoU.</b>

## Instance Segmentation Module
<table>
<thead>
<tr>
<th>Model Name</th>
<th>Mask AP</th>
<th>Model Size (M)</th>
</tr>
</thead>
<tbody>
<tr>
<td>Mask-RT-DETR-H</td>
<td>50.6</td>
<td>449.9 M</td>
</tr>
<tr>
<td>Mask-RT-DETR-L</td>
<td>45.7</td>
<td>113.6 M</td>
</tr>
<tr>
<td>Mask-RT-DETR-M</td>
<td>42.7</td>
<td>66.6 M</td>
</tr>
<tr>
<td>Cascade-MaskRCNN-ResNet50-FPN</td>
<td>36.3</td>
<td>254.8 M</td>
</tr>
<tr>
<td>Cascade-MaskRCNN-ResNet50-vd-SSLDv2-FPN</td>
<td>39.1</td>
<td>254.7 M</td>
</tr>
<tr>
<td>PP-YOLOE_seg-S</td>
<td>32.5</td>
<td>31.5 M</td>
</tr>
</tbody>
</table>
<b>Note: The above accuracy metrics are for</b> [COCO2017](https://cocodataset.org/#home) <b>validation set Mask AP(0.5:0.95).</b>

## Text Detection Module
<table>
<thead>
<tr>
<th>Model Name</th>
<th>Detection Hmean (%)</th>
<th>Model Size (M)</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-OCRv4_mobile_det</td>
<td>77.79</td>
<td>4.2 M</td>
</tr>
<tr>
<td>PP-OCRv4_server_det</td>
<td>82.69</td>
<td>100.1 M</td>
</tr>
</tbody>
</table>
<b>Note: The above accuracy metrics are evaluated on PaddleOCR's self-built Chinese dataset, covering street scenes, web images, documents, and handwritten scenarios, with 500 images for detection.</b>

## Text Recognition Module
<table>
<thead>
<tr>
<th>Model Name</th>
<th>Recognition Avg Accuracy (%)</th>
<th>Model Size (M)</th>
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
<b>Note: The above accuracy metrics are evaluated on PaddleOCR's self-built Chinese dataset, covering street scenes, web images, documents, and handwritten scenarios, with 11,000 images for text recognition.</b>

<table>
<thead>
<tr>
<th>Model Name</th>
<th>Recognition Avg Accuracy (%)</th>
<th>Model Size (M)</th>
</tr>
</thead>
<tbody>
<tr>
<td>ch_SVTRv2_rec</td>
<td>68.81</td>
<td>73.9 M</td>
</tr>
</tbody>
</table>
<b>Note: The above accuracy metrics are evaluated on the [PaddleOCR Algorithm Model Challenge - Task 1: OCR End-to-End Recognition](https://aistudio.baidu.com/competition/detail/1131/0/introduction) A-Rank.</b>

<table>
<thead>
<tr>
<th>Model Name</th>
<th>Recognition Avg Accuracy (%)</th>
<th>Model Size (M)</th>
</tr>
</thead>
<tbody>
<tr>
<td>ch_RepSVTR_rec</td>
<td>65.07</td>
<td>22.1 M</td>
</tr>
</tbody>
</table>
<b>Note: The above accuracy metrics are evaluated on the [PaddleOCR Algorithm Model Challenge - Task 1: OCR End-to-End Recognition](https://aistudio.baidu.com/competition/detail/1131/0/introduction) B-Rank.</b>

## Table Structure Recognition Module
<table>
<thead>
<tr>
<th>Model Name</th>
<th>Accuracy (%)</th>
<th>Model Size (M)</th>
</tr>
</thead>
<tbody>
<tr>
<td>SLANet</td>
<td>76.31</td>
<td>6.9 M</td>
</tr>
</tbody>
</table>
<b>Note: The above accuracy metrics are measured on the PubtabNet English table recognition dataset.</b>

## Layout Analysis Module
<table>
<thead>
<tr>
<th>Model Name</th>
<th>mAP (%)</th>
<th>Model Size (M)</th>
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
<b>Note: The evaluation set for the above accuracy metrics is PaddleOCR's self-built layout analysis dataset, containing 10,000 images.</b>

## Time Series Forecasting Module
<table>
<thead>
<tr>
<th>Model Name</th>
<th>MSE</th>
<th>MAE</th>
<th>Model Size (M)</th>
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
<td>Nonstationary</td>
<td>0.600</td>
<td>0.515</td>
<td>55.5 M</td>
</tr>
<tr>
<td>PatchTST</td>
<td>0.385</td>
<td>0.397</td>
<td>2.0M</td>
</tr>
<tr>
<td>RLinear</td>
<td>0.384</td>
<td>0.392</td>
<td>40K</td>
</tr>
<tr>
<td>TiDE</td>
<td>0.405</td>
<td>0.412</td>
<td>31.7M</td>
</tr>
<tr>
<td>TimesNet</td>
<td>0.417</td>
<td>0.431</td>
<td>4.9M</td>
</tr>
</tbody>
</table>
<b>Note: The above accuracy metrics are measured on the [ETTH1](https://paddle-model-ecology.bj.bcebos.com/paddlex/data/Etth1.tar) dataset (evaluation results on the test set test.csv).</b>

## Time Series Anomaly Detection Module
<table>
<thead>
<tr>
<th>Model Name</th>
<th>Precision</th>
<th>Recall</th>
<th>F1-Score</th>
<th>Model Size (M)</th>
</tr>
</thead>
<tbody>
<tr>
<td>AutoEncoder_ad</td>
<td>99.36</td>
<td>84.36</td>
<td>91.25</td>
<td>52K</td>
</tr>
<tr>
<td>DLinear_ad</td>
<td>98.98</td>
<td>93.96</td>
<td>96.41</td>
<td>112K</td>
</tr>
<tr>
<td>Nonstationary_ad</td>
<td>98.55</td>
<td>88.95</td>
<td>93.51</td>
<td>1.8M</td>
</tr>
<tr>
<td>PatchTST_ad</td>
<td>98.78</td>
<td>90.70</td>
<td>94.57</td>
<td>320K</td>
</tr>
<tr>
<td>TimesNet_ad</td>
<td>98.37</td>
<td>94.80</td>
<td>96.56</td>
<td>1.3M</td>
</tr>
</tbody>
</table>
<b>Note: The above accuracy metrics are measured on the [PSM](https://paddle-model-ecology.bj.bcebos.com/paddlex/data/ts_anomaly_examples.tar) dataset.</b>

## Time Series Classification Module
<table>
<thead>
<tr>
<th>Model Name</th>
<th>Acc (%)</th>
<th>Model Size (M)</th>
</tr>
</thead>
<tbody>
<tr>
<td>TimesNet_cls</td>
<td>87.5</td>
<td>792K</td>
</tr>
</tbody>
</table>
<b>Note: The above accuracy metrics are measured on the UWaveGestureLibrary: [Training](https://paddlets.bj.bcebos.com/classification/UWaveGestureLibrary_TRAIN.csv), [Evaluation](https://paddlets.bj.bcebos.com/classification/UWaveGestureLibrary_TEST.csv) datasets.</b>
