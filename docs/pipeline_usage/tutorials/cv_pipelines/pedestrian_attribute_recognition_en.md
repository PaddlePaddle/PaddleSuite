English | [简体中文](pedestrian_attribute_recognition.md)

# Pedestrian Attribute Recognition Pipeline Tutorial

## 1. Introduction to Pedestrian Attribute Recognition Pipeline
Pedestrian attribute recognition is a key function in computer vision systems, used to locate and label specific characteristics of pedestrians in images or videos, such as gender, age, clothing color, and style. This task not only requires accurately detecting pedestrians but also identifying detailed attribute information for each pedestrian. The pedestrian attribute recognition pipeline is an end-to-end serial system for locating and recognizing pedestrian attributes, widely used in smart cities, security surveillance, and other fields, significantly enhancing the system's intelligence level and management efficiency.

![](https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/pipelines/pedestrian_attribute_recognition/01.jpg)

**The pedestrian attribute recognition pipeline includes a pedestrian detection module and a pedestrian attribute recognition module**, with several models in each module. Which models to use specifically can be selected based on the benchmark data below. **If you prioritize model accuracy, choose models with higher accuracy; if you prioritize inference speed, choose models with faster inference; if you prioritize model storage size, choose models with smaller storage**.

<details>
   <summary> 👉Model List Details</summary>

**Pedestrian Detection Module**:

<table>
  <tr>
    <th>Model</th>
    <th>mAP(0.5:0.95)</th>
    <th>mAP(0.5)</th>
    <th>GPU Inference Time (ms)</th>
    <th>CPU Inference Time (ms)</th>
    <th>Model Size (M)</th>
    <th>Description</th>
  </tr>
  <tr>
    <td>PP-YOLOE-L_human</td>
    <td>48.0</td>
    <td>81.9</td>
    <td>32.8</td>
    <td>777.7</td>
    <td>196.02</td>
    <td rowspan="2">Pedestrian detection model based on PP-YOLOE</td>
  </tr>
  <tr>
    <td>PP-YOLOE-S_human</td>
    <td>42.5</td>
    <td>77.9</td>
    <td>15.0</td>
    <td>179.3</td>
    <td>28.79</td>
  </tr>
</table>

**Note: The above accuracy metrics are mAP(0.5:0.95) on the CrowdHuman dataset. All model GPU inference times are based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speeds are based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.**

**Pedestrian Attribute Recognition Module**:

|Model|mA (%)|GPU Inference Time (ms)|CPU Inference Time (ms)|Model Size (M)|Description|
|-|-|-|-|-|-|
|PP-LCNet_x1_0_pedestrian_attribute|92.2|3.84845|9.23735|6.7 M|PP-LCNet_x1_0_pedestrian_attribute is a lightweight pedestrian attribute recognition model based on PP-LCNet, covering 26 categories.|

**Note: The above accuracy metrics are mA on PaddleX's internally built dataset. GPU inference times are based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speeds are based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.**

</details>

## 2. Quick Start
The pre-trained model pipelines provided by PaddleX can quickly demonstrate their effectiveness. You can experience the pedestrian attribute recognition pipeline online or locally using command line or Python.

### 2.1 Online Experience
Not supported yet.

### 2.2 Local Experience
Before using the pedestrian attribute recognition pipeline locally, ensure you have completed the installation of the PaddleX wheel package following the [PaddleX Local Installation Tutorial](../../../installation/installation.md).

#### 2.2.1 Command Line Experience
You can quickly experience the pedestrian attribute recognition pipeline with a single command. Use the [test file](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/pedestrian_attribute_002.jpg) and replace `--input` with the local path for prediction.

```bash
paddlex --pipeline pedestrian_attribute_recognition --input pedestrian_attribute_002.jpg --device gpu:0
```
Parameter Description:

```
--pipeline: The name of the pipeline, here it is the pedestrian attribute recognition pipeline.
--input: The local path or URL of the input image to be processed.
--device: The GPU index to use (e.g., gpu:0 means using the first GPU, gpu:1,2 means using the second and third GPUs), or you can choose to use CPU (--device cpu).
```

When executing the above Python script, the default pedestrian attribute recognition pipeline configuration file is loaded. If you need a custom configuration file, you can run the following command to obtain it:

<details>
   <summary> 👉Click to Expand</summary>

```
paddlex --get_pipeline_config pedestrian_attribute_recognition
```
After execution, the pedestrian attribute recognition pipeline configuration file will be saved in the current path. If you wish to specify a custom save location, you can run the following command (assuming the custom save location is `./my_path`):

```
paddlex --get_pipeline_config pedestrian_attribute_recognition --save_path ./my_path
```

After obtaining the pipeline configuration file, you can replace `--pipeline` with the saved path of the configuration file to make it effective. For example, if the configuration file is saved at `./pedestrian_attribute_recognition.yaml`, simply execute:

```bash
paddlex --pipeline ./pedestrian_attribute_recognition.yaml --input pedestrian_attribute_002.jpg --device gpu:0
```
Among them, parameters such as `--model` and `--device` do not need to be specified, and the parameters in the configuration file will be used. If parameters are still specified, the specified parameters will take precedence.

</details>

#### 2.2.2 Python Script Integration
A few lines of code are sufficient for quick inference of the pipeline. Taking the pedestrian attribute recognition pipeline as an example:

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="pedestrian_attribute_recognition")

output = pipeline.predict("pedestrian_attribute_002.jpg")
for res in output:
    res.print()  ## Print the structured output of the prediction
    res.save_to_img("./output/")  ## Save the visualized image of the result
    res.save_to_json("./output/")  ## Save the structured output of the prediction
```
The results obtained are the same as those from the command line approach.

In the above Python script, the following steps are executed:

(1) Instantiate the `create_pipeline` to create a pipeline object: Specific parameter descriptions are as follows:

| Parameter | Description | Parameter Type | Default Value |
|-----------|-------------|----------------|---------------|
| `pipeline` | The name of the pipeline or the path to the pipeline configuration file. If it is the name of the pipeline, it must be a pipeline supported by PaddleX. | `str` | None |
| `device` | The device for pipeline model inference. Supports: "gpu", "cpu". | `str` | "gpu" |
| `use_hpip` | Whether to enable high-performance inference, only available when the pipeline supports high-performance inference. | `bool` | `False` |

(2) Call the `predict` method of the pedestrian attribute recognition pipeline object for inference prediction: The `predict` method parameter is `x`, which is used to input data to be predicted, supporting multiple input methods. Specific examples are as follows:

| Parameter Type | Description |
|----------------|-----------------------------------------------------------------------------------------------------------|
| Python Var | Supports directly passing in Python variables, such as image data represented by numpy.ndarray. |
| `str` | Supports passing in the file path of the data to be predicted, such as the local path of an image file: `/root/data/img.jpg`. |
| `str` | Supports passing in the URL of the data file to be predicted, such as the network URL of an image file: [Example](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/pedestrian_attribute_002.jpg). |
| `str` | Supports passing in a local directory, which should contain the data files to be predicted, such as the local path: `/root/data/`. |
| `dict` | Supports passing in a dictionary type, where the key needs to correspond to the specific task, such as "img" for the pedestrian attribute recognition task, and the value of the dictionary supports the above data types, for example: `{"img": "/root/data1"}`. |
| `list` | Supports passing in a list, where the elements of the list need to be the above data types, such as `[numpy.ndarray, numpy.ndarray], ["/root/data/img1.jpg", "/root/data/img2.jpg"], ["/root/data1", "/root/data2"], [{"img": "/root/data1"}, {"img": "/root/data2/img.jpg"}]`. |

(3) Obtain the prediction results by calling the `predict` method: The `predict` method is a `generator`, so prediction results need to be obtained through iteration. The `predict` method predicts data in batches, so the prediction results are in the form of a list.

(4) Process the prediction results: The prediction result for each sample is of type `dict` and supports printing or saving as a file. The supported save types are related to the specific pipeline, such as:

| Method | Description | Method Parameters |
|--------|-------------|-------------------|
| `print` | Print the results to the terminal | `- format_json`: bool, whether to format the output content with json indentation, default is True;<br>`- indent`: int, json formatting setting, only effective when `format_json` is True, default is 4;<br>`- ensure_ascii`: bool, json formatting setting, only effective when `format_json` is True, default is False; |
| `save_to_json` | Save the results as a json-formatted file | `- save_path`: str, the path to save the file, when it is a directory, the saved file name is consistent with the input file name;<br>`- indent`: int, json formatting setting, default is 4;<br>`- ensure_ascii`: bool, json formatting setting, default is False; |
| `save_to_img` | Save the results as an image file |

If you have obtained the configuration file, you can customize various configurations for the pedestrian attribute recognition pipeline by simply modifying the `pipeline` parameter in the `create_pipeline` method to the path of your pipeline configuration file.

For example, if your configuration file is saved as `./my_path/pedestrian_attribute_recognition*.yaml`, you only need to execute:

```python
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="./my_path/pedestrian_attribute_recognition.yaml")
output = pipeline.predict("pedestrian_attribute_002.jpg")
for res in output:
    res.print()  # Print the structured output of the prediction
    res.save_to_img("./output/")  # Save the visualized result image
    res.save_to_json("./output/")  # Save the structured output of the prediction
```
## 3. Development Integration/Deployment
If the face recognition pipeline meets your requirements for inference speed and accuracy, you can proceed directly with development integration/deployment.

If you need to directly apply the face recognition pipeline in your Python project, you can refer to the example code in [2.2.2 Python Script Integration](#222-python-script-integration).

Additionally, PaddleX provides three other deployment methods, detailed as follows:

🚀 **High-Performance Inference**: In actual production environments, many applications have stringent standards for the performance metrics of deployment strategies (especially response speed) to ensure efficient system operation and smooth user experience. To this end, PaddleX provides high-performance inference plugins aimed at deeply optimizing model inference and pre/post-processing to significantly speed up the end-to-end process. For detailed high-performance inference procedures, please refer to the [PaddleX High-Performance Inference Guide](../../../pipeline_deploy/high_performance_inference.md).

☁️ **Service-Oriented Deployment**: Service-oriented deployment is a common deployment form in actual production environments. By encapsulating inference functionality as services, clients can access these services through network requests to obtain inference results. PaddleX supports users in achieving service-oriented deployment of pipelines at low cost. For detailed service-oriented deployment procedures, please refer to the [PaddleX Service-Oriented Deployment Guide](../../../pipeline_deploy/service_deploy.md).

Below are the API reference and multi-language service invocation examples:

<details>
<summary>API Reference</summary>

For all operations provided by the service:

- The response body and the request body of POST requests are both JSON data (JSON objects).
- When the request is successfully processed, the response status code is `200`, and the attributes of the response body are as follows:

    | Name | Type | Meaning |
    |-|-|-|
    |`errorCode`|`integer`|Error code. Fixed to `0`. |
    |`errorMsg`|`string`|Error description. Fixed to `"Success"`. |

    The response body may also have a `result` attribute of type `object`, which stores the operation result information.

- When the request is not successfully processed, the attributes of the response body are as follows:

    | Name | Type | Meaning |
    |-|-|-|
    |`errorCode`|`integer`|Error code. Same as the response status code. |
    |`errorMsg`|`string`|Error description. |

The operations provided by the service are as follows:

- **`infer`**

    Get pedestrian attribute recognition results.

    `POST /pedestrian-attribute-recognition`

    - The request body properties are as follows:

        | Name | Type | Description | Required |
        |------|------|-------------|----------|
        | `image` | `string` | The URL of an image file accessible by the service or the Base64 encoded result of the image file content. | Yes |

    - When the request is processed successfully, the `result` of the response body has the following properties:

        | Name | Type | Description |
        |------|------|-------------|
        | `pedestrians` | `array` | Information about the pedestrian's location and attributes. |
        | `image` | `string` | The pedestrian attribute recognition result image. The image is in JPEG format and encoded using Base64. |

        Each element in `pedestrians` is an `object` with the following properties:

        | Name | Type | Description |
        |------|------|-------------|
        | `bbox` | `array` | The location of the pedestrian. The elements in the array are the x-coordinate of the top-left corner, the y-coordinate of the top-left corner, the x-coordinate of the bottom-right corner, and the y-coordinate of the bottom-right corner of the bounding box, respectively. |
        | `score` | `number` | The detection score. |
        | `attributes` | `array` | The pedestrian attributes. |

        Each element in `attributes` is an `object` with the following properties:

        | Name | Type | Description |
        |------|------|-------------|
        | `label` | `string` | The label of the attribute. |
        | `score` | `number` | The classification score. |

</details>

<details>
<summary>Multi-Language Service Invocation Examples</summary>

<details>
<summary>Python</summary>

```python
import base64
import requests

API_URL = "http://localhost:8080/pedestrian-attribute-recognition"
image_path = "./demo.jpg"
output_image_path = "./out.jpg"

with open(image_path, "rb") as file:
    image_bytes = file.read()
    image_data = base64.b64encode(image_bytes).decode("ascii")

payload = {"image": image_data}

response = requests.post(API_URL, json=payload)

assert response.status_code == 200
result = response.json()["result"]
with open(output_image_path, "wb") as file:
    file.write(base64.b64decode(result["image"]))
print(f"Output image saved at {output_image_path}")
print("\nDetected pedestrians:")
print(result["pedestrians"])
```

</details>
</details>
<br/>
<br/>

📱 **Edge Deployment**: Edge deployment is a method where computing and data processing functions are placed on the user's device itself, allowing the device to process data directly without relying on remote servers. PaddleX supports deploying models on edge devices such as Android. For detailed edge deployment procedures, please refer to the [PaddleX Edge Deployment Guide](../../../pipeline_deploy/edge_deploy_en.md).
You can choose an appropriate method to deploy your model pipeline based on your needs, and proceed with subsequent AI application integration.


## 4. Custom Development
If the default model weights provided by the Face Recognition Pipeline do not meet your expectations in terms of accuracy or speed for your specific scenario, you can try to further **fine-tune** the existing models using **your own domain-specific or application-specific data** to enhance the recognition performance of the pipeline in your scenario.

### 4.1 Model Fine-tuning
Since the Face Recognition Pipeline consists of two modules (face detection and face recognition), the suboptimal performance of the pipeline may stem from either module.

You can analyze images with poor recognition results. If you find that many faces are not detected during the analysis, it may indicate deficiencies in the face detection model. In this case, you need to refer to the [Custom Development](../../../module_usage/tutorials/cv_modules/face_detection_en.md#IV.-Custom-Development) section in the [Face Detection Module Development Tutorial](../../../module_usage/tutorials/cv_modules/face_detection_en.md) and use your private dataset to fine-tune the face detection model. If matching errors occur in detected faces, it suggests that the face feature model needs further improvement. You should refer to the [Custom Development](../../../module_usage/tutorials/cv_modules/face_feature_en.md#IV.-Custom-Development) section in the [Face Feature Module Development Tutorial](../../../module_usage/tutorials/cv_modules/face_feature_en.md) to fine-tune the face feature model.

### 4.2 Model Application
After completing fine-tuning training with your private dataset, you will obtain local model weight files.

To use the fine-tuned model weights, you only need to modify the pipeline configuration file by replacing the local paths of the fine-tuned model weights with the corresponding paths in the pipeline configuration file:

```bash

......
Pipeline:
  device: "gpu:0"
  det_model: "BlazeFace"        # Can be modified to the local path of the fine-tuned face detection model
  rec_model: "MobileFaceNet"    # Can be modified to the local path of the fine-tuned face recognition model
  det_batch_size: 1
  rec_batch_size: 1
  device: gpu
......
```
Subsequently, refer to the command-line method or Python script method in [2.2 Local Experience](#22-Local-Experience) to load the modified pipeline configuration file.
Note: Currently, setting separate `batch_size` for face detection and face recognition models is not supported.

## 5. Multi-hardware Support
PaddleX supports various mainstream hardware devices such as NVIDIA GPUs, Kunlun XPU, Ascend NPU, and Cambricon MLU. **Simply modifying the `--device` parameter** allows seamless switching between different hardware.

For example, when running the face recognition pipeline using Python and changing the running device from an NVIDIA GPU to an Ascend NPU, you only need to modify the `device` in the script to `npu`:

```python
from paddlex import create_pipeline

pipeline = create_pipeline(
    pipeline="face_recognition",
    device="npu:0" # gpu:0 --> npu:0
)
```
If you want to use the face recognition pipeline on more types of hardware, please refer to the [PaddleX Multi-device Usage Guide](../../../other_devices_support/multi_devices_use_guide_en.md).
