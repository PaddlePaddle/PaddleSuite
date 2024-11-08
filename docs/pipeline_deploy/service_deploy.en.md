---
comments: true
---

# PaddleX Serving Deployment Guide

Serving deployment is a common form of deployment in real-world production environments. By encapsulating inference capabilities as services, clients can access these services through network requests to obtain inference results. PaddleX enables users to achieve low-cost serving deployment for production lines. This document will first introduce the basic process of serving deployment using PaddleX, followed by considerations and potential operations when using the service in a production environment.

<b>Note</b>
- <b>Serving deployment provides services for model pipelines, not specific to individual pipeline modules.</b>

Serving Deployment Example Diagram:

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipeline_deploy/serving_en.png"  width="300" />

## 1. Basic Process

### 1.1 Install the Serving Deployment Plugin

Execute the following command to install the serving deployment plugin:

```shell
paddlex --install serving
```

### 1.2 Start the Service

Start the service through the PaddleX CLI with the following command format:

```shell
paddlex --serve --pipeline {pipeline_name_or_path} [{other_command_line_options}]
```

Taking the General Image Classification Pipeline as an example:

```shell
paddlex --serve --pipeline image_classification
```

After the service starts successfully, you will see information similar to the following:

```
INFO:     Started server process [63108]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)
```

`--pipeline` can be specified as an official pipeline name or the path to a local pipeline configuration file. PaddleX uses this to build the pipeline and deploy it as a service. To adjust configurations (such as model path, batch_size, deployment device), please refer to the <b>"Model Application"</b> section in the [General Image Classification Pipeline Tutorial](../pipeline_usage/tutorials/cv_pipelines/image_classification.en.md) (for other pipelines, refer to the corresponding tutorials in the <b>"1.3 Calling the Service"</b> table).

Command-line options related to serving deployment are as follows:

<table>
<thead>
<tr>
<th>Name</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>--pipeline</code></td>
<td>Pipeline name or pipeline configuration file path.</td>
</tr>
<tr>
<td><code>--device</code></td>
<td>Deployment device for the pipeline. Defaults to <code>cpu</code> (If GPU is unavailable) or <code>gpu</code> (If GPU is available).</td>
</tr>
<tr>
<td><code>--host</code></td>
<td>Hostname or IP address bound to the server. Defaults to 0.0.0.0.</td>
</tr>
<tr>
<td><code>--port</code></td>
<td>Port number listened to by the server. Defaults to 8080.</td>
</tr>
<tr>
<td><code>--use_hpip</code></td>
<td>Enables the high-performance inference plugin if specified.</td>
</tr>
<tr>
<td><code>--serial_number</code></td>
<td>Serial number used by the high-performance inference plugin. Only valid when the high-performance inference plugin is enabled. Note that not all pipelines and models support the use of the high-performance inference plugin. For detailed support, please refer to the <a href="./high_performance_inference.en.md">PaddleX High-Performance Inference Guide</a>.</td>
</tr>
<tr>
<td><code>--update_license</code></td>
<td>Activates the license online if specified. Only valid when the high-performance inference plugin is enabled.</td>
</tr>
</tbody>
</table>
</table>

### 1.3 Call the Service

Please refer to the <b>"Development Integration/Deployment"</b> section in the usage tutorials for each pipeline.

<table>
<thead>
<tr>
<th>Model Pipelines</th>
<th>Usage Tutorials</th>
</tr>
</thead>
<tbody>
<tr>
<td>General Image Classification Pipeline</td>
<td><a href="../pipeline_usage/tutorials/cv_pipelines/image_classification.en.md">Tutorial for Using the General Image Classification Pipeline</a></td>
</tr>
<tr>
<td>General Object Detection Pipeline</td>
<td><a href="../pipeline_usage/tutorials/cv_pipelines/object_detection.en.md">Tutorial for Using the General Object Detection Pipeline</a></td>
</tr>
<tr>
<td>General Semantic Segmentation Pipeline</td>
<td><a href="../pipeline_usage/tutorials/cv_pipelines/semantic_segmentation.en.md">Tutorial for Using the General Semantic Segmentation Pipeline</a></td>
</tr>
<tr>
<td>General Instance Segmentation Pipeline</td>
<td><a href="../pipeline_usage/tutorials/cv_pipelines/instance_segmentation.en.md">Tutorial for Using the General Instance Segmentation Pipeline</a></td>
</tr>
<tr>
<td>General Image Multi-Label Classification Pipeline</td>
<td><a href="../pipeline_usage/tutorials/cv_pipelines/image_multi_label_classification.en.md">Tutorial for Using the General Image Multi-Label Classification Pipeline</a></td>
</tr>
<tr>
<td>Small Object Detection Pipeline</td>
<td><a href="../pipeline_usage/tutorials/cv_pipelines/small_object_detection.en.md">Tutorial for Using the Small Object Detection Pipeline</a></td>
</tr>
<tr>
<td>Image Anomaly Detection Pipeline</td>
<td><a href="../pipeline_usage/tutorials/cv_pipelines/image_anomaly_detection.en.md">Tutorial for Using the Image Anomaly Detection Pipeline</a></td>
</tr>
<tr>
<td>General OCR Pipeline</td>
<td><a href="../pipeline_usage/tutorials/ocr_pipelines/OCR.en.md">Tutorial for Using the General OCR Pipeline</a></td>
</tr>
<tr>
<td>General Table Recognition Pipeline</td>
<td><a href="../pipeline_usage/tutorials/ocr_pipelines/table_recognition.en.md">Tutorial for Using the General Table Recognition Pipeline</a></td>
</tr>
<tr>
<td>General Layout Parsing Pipeline</td>
<td><a href="../pipeline_usage/tutorials/ocr_pipelines/layout_parsing.en.md">Tutorial for Using the Layout Parsing Recognition Pipeline</a></td>
</tr>
<tr>
<td>Formula Recognition Pipeline</td>
<td><a href="../pipeline_usage/tutorials/ocr_pipelines/formula_recognition.en.md">Tutorial for Using the Formula Recognition Pipeline</a></td>
</tr>
<tr>
<td>Seal Text Recognition Pipeline</td>
<td><a href="../pipeline_usage/tutorials/ocr_pipelines/seal_recognition.en.md">Tutorial for Using the Seal Text Recognition Pipeline</a></td>
</tr>
<tr>
<td>Time Series Forecasting Pipeline</td>
<td><a href="../pipeline_usage/tutorials/time_series_pipelines/time_series_forecasting.en.md">Tutorial for Using the Time Series Forecasting Pipeline</a></td>
</tr>
<tr>
<td>Time Series Anomaly Detection Pipeline</td>
<td><a href="../pipeline_usage/tutorials/time_series_pipelines/time_series_anomaly_detection.en.md">Tutorial for Using the Time Series Anomaly Detection Pipeline</a></td>
</tr>
<tr>
<td>Time Series Classification Pipeline</td>
<td><a href="../pipeline_usage/tutorials/time_series_pipelines/time_series_classification.en.md">Tutorial for Using the Time Series Classification Pipeline</a></td>
</tr>
<tr>
<td>Document Scene Information Extraction v3 Pipeline</td>
<td><a href="../pipeline_usage/tutorials/information_extraction_pipelines/document_scene_information_extraction.en.md">Tutorial for Using the Document Scene Information Extraction v3 Pipeline</a></td>
</tr>
</tbody>
</table>
## 2. Deploy Services for Production

When deploying services into production environments, the stability, efficiency, and security of the services are of paramount importance. Below are some recommendations for deploying services into production.

### 2.1 Utilize PaddleX high-performance inference Plugin

In scenarios where strict response time requirements are imposed on applications, the PaddleX high-performance inference Plugin can be used to accelerate model inference and pre/post-processing, thereby reducing response time and increasing throughput.

To use the PaddleX high-performance inference Plugin, please refer to the [PaddleX High-Performance Inference Guide](./high_performance_inference.en.md) for installing the high-performance inference plugin, obtaining serial numbers, and activating the plugin. Additionally, not all pipelines, models, and environments support the use of the high-performance inference plugin. For detailed support information, please refer to the section on pipelines and models that support the high-performance inference plugin.

When starting the PaddleX pipeline service, you can specify `--use_hpip` along with the serial number to use the high-performance inference plugin. If you wish to perform online activation, you should also specify `--update_license`. Example usage:

```bash
paddlex --serve --pipeline image_classification --use_hpip --serial_number {serial_number}

# If you wish to perform online activation
paddlex --serve --pipeline image_classification --use_hpip --serial_number {serial_number} --update_license
```

### 2.2 Consider Security

A typical scenario involves an application accepting inputs from the network, with the PaddleX pipeline service acting as a module within the application, interacting with other modules through APIs. In this case, the position of the PaddleX pipeline service within the application is crucial. The service-oriented deployment solution provided by PaddleX focuses on efficiency and ease of use but does not perform sufficient security checks on request bodies. Malicious requests from the network, such as excessively large images or carefully crafted data, can lead to severe consequences like service crashes. Therefore, it is recommended to place the PaddleX pipeline service within the application's internal network, avoiding direct processing of external inputs, and ensuring it only processes trustworthy requests. Appropriate protective measures, such as input validation and authentication, should be added at the application's outer layer.
