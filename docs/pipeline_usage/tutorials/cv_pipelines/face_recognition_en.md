English | [简体中文](face_recognition.md)

# Face Recognition Pipeline Tutorial

## 1. Introduction to the Face Recognition Pipeline
Face recognition is a crucial component in the field of computer vision, aiming to automatically identify individuals by analyzing and comparing facial features. This task involves not only detecting faces in images but also extracting and matching facial features to find corresponding identity information in a database. Face recognition is widely used in security authentication, surveillance systems, social media, smart devices, and other scenarios.

The face recognition pipeline is an end-to-end system dedicated to solving face detection and recognition tasks. It can quickly and accurately locate face regions in images, extract facial features, and retrieve and compare them with pre-established features in a feature database to confirm identity information.

![](https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/face_recognition/01.png)

**The face recognition pipeline includes a face detection module and a face feature module**, with several models in each module. Which models to use can be selected based on the benchmark data below. **If you prioritize model accuracy, choose models with higher accuracy; if you prioritize inference speed, choose models with faster inference; if you prioritize model size, choose models with smaller storage requirements**.

<details>
   <summary> 👉Model List Details</summary>

**Face Detection Module**:

| Model | AP (%)<br>Easy/Medium/Hard | GPU Inference Time (ms) | CPU Inference Time | Model Size (M) | Description |
|--------------------------|-----------------|--------------|---------|------------|-----------------------------|
| BlazeFace                | 77.7/73.4/49.5  |              |         | 0.447      | A lightweight and efficient face detection model |
| BlazeFace-FPN-SSH        | 83.2/80.5/60.5  |              |         | 0.606      | Improved BlazeFace with FPN and SSH structures |
| PicoDet_LCNet_x2_5_face	 | 93.7/90.7/68.1  |              |         | 28.9       | Face detection model based on PicoDet_LCNet_x2_5 |
| PP-YOLOE_plus-S_face     | 93.9/91.8/79.8  |              |         | 26.5       | Face detection model based on PP-YOLOE_plus-S |

Note: The above accuracy metrics are evaluated on the WIDER-FACE validation set with an input size of 640x640. All GPU inference times are based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speeds are based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.

**Face Recognition Module**:

| Model           | Output Feature Dimension | AP (%)<br>AgeDB-30/CFP-FP/LFW | GPU Inference Time (ms) | CPU Inference Time | Model Size (M) | Description |
|---------------|--------|-------------------------------|--------------|---------|------------|-------------------------------------|
| MobileFaceNet | 128    | 96.28/96.71/99.58             |              |         | 4.1        | Face recognition model trained on MS1Mv3 based on MobileFaceNet |
| ResNet50_face      | 512    | 98.12/98.56/99.77             |              |         | 87.2       | Face recognition model trained on MS1Mv3 based on ResNet50 |

Note: The above accuracy metrics are Accuracy scores measured on the AgeDB-30, CFP-FP, and LFW datasets, respectively. All GPU inference times are based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speeds are based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.

</details>

## 2. Quick Start
The pre-trained model pipelines provided by PaddleX can be quickly experienced. You can experience the effects of the face recognition pipeline online or locally using command-line or Python.

### 2.1 Online Experience

Oneline Experience is not supported at the moment.

### 2.2 Local Experience
> ❗ Before using the facial recognition pipeline locally, please ensure that you have completed the installation of the PaddleX wheel package according to the [PaddleX Installation Guide](../../../installation/installation.md).

#### 2.2.1 Command Line Experience

Command line experience is not supported at the moment.
#### 2.2.2 Integration via Python Script

Please download the [test image](https://paddle-model-ecology.bj.bcebos.com/paddlex/demo_data/friends1.jpg) for testing. In the example of running this pipeline, you need to pre-build a facial feature library. You can refer to the following instructions to download the official demo data to be used for subsequent construction of the facial feature library. You can use the following command to download the demo dataset to a specified folder:

```bash
cd /path/to/paddlex
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/face_demo_gallery.tar
tar -xf ./face_demo_gallery.tar
```

If you wish to build a facial feature library using a private dataset, please refer to [Section 2.3: Data Organization for Building a Feature Library](#23-data-organization-for-building-a-feature-library). Afterward, you can complete the establishment of the facial feature library and quickly perform inference with the facial recognition pipeline using just a few lines of code.

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="face_recognition")

pipeline.build_index(data_root="face_demo_gallery", index_dir="face_gallery_index")

output = pipeline.predict("friends1.jpg")
for res in output:
    res.print()
    res.save_to_img("./output/")
```

In the above Python script, the following steps are executed:

(1) Instantiate the `create_pipeline` to create a face recognition pipeline object. The specific parameter descriptions are as follows:

| Parameter | Description | Type | Default |
|-|-|-|-|
| `pipeline` | The name of the pipeline or the path to the pipeline configuration file. If it is the pipeline name, it must be a pipeline supported by PaddleX. | `str` | None |
| `device` | The device for pipeline model inference. Supports: "gpu", "cpu". | `str` | "gpu" |
| `use_hpip` | Whether to enable high-performance inference, only available when the pipeline supports high-performance inference. | `bool` | `False` |

(2) Call the `build_index` method of the face recognition pipeline object to build the facial feature library. The specific parameters are described as follows:

| Parameter | Description | Type | Default |
|-|-|-|-|
| `data_root` | The root directory of the dataset, with data organization referring to [Section 2.3: Data Organization for Building a Feature Library](#2.3-Data-Organization-for-Building-a-Feature-Library) | `str` | None |
| `index_dir` | The save path for the feature library. After successfully calling the `build_index` method, two files will be generated in this path:<br> `"id_map.pkl"` saves the mapping relationship between image IDs and image feature labels;<br> `"vector.index"` stores the feature vectors of each image. | `str` | None |

(3) Call the `predict` method of the face recognition pipeline object for inference prediction: The `predict` method parameter is `x`, used to input data to be predicted, supporting multiple input methods, as shown in the following examples:

| Parameter Type | Description |
|----------------|-----------------------------------------------------------------------------------------------------------|
| Python Var | Supports directly passing in Python variables, such as image data represented by `numpy.ndarray`. |
| `str` | Supports passing in the file path of the data to be predicted, such as the local path of an image file: `/root/data/img.jpg`. |
| `str` | Supports passing in the URL of the data file to be predicted, such as the network URL of an image file: [Example](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_001.png). |
| `str` | Supports passing in a local directory containing the data files to be predicted, such as the local path: `/root/data/`. |
| `dict` | Supports passing in a dictionary type, where the key needs to correspond to the specific task, such as "img" for image classification tasks, and the value of the dictionary supports the above types of data, for example: `{"img": "/root/data1"}`. |
| `list` | Supports passing in a list, where the list elements need to be the above types of data, such as `[numpy.ndarray, numpy.ndarray], ["/root/data/img1.jpg", "/root/data/img2.jpg"], ["/root/data1", "/root/data2"], [{"img": "/root/data1"}, {"img": "/root/data2/img.jpg"}]`. |

(4) Obtain the prediction results by calling the `predict` method: The `predict` method is a `generator`, so prediction results need to be obtained through iteration. The `predict` method predicts data in batches, so the prediction results are in the form of a list.

(5) Process the prediction results: The prediction result for each sample is of type `dict`, and it supports printing or saving to a file. The supported file types depend on the specific pipeline, such as:

| Method       | Description                 | Method Parameters                                                                                      |
|--------------|-----------------------------|--------------------------------------------------------------------------------------------------------|
| print        | Print results to the terminal | `- format_json`: Boolean, whether to format the output with JSON indentation, default is True; <br>`- indent`: Integer, JSON formatting setting, effective only when format_json is True, default is 4; <br>`- ensure_ascii`: Boolean, JSON formatting setting, effective only when format_json is True, default is False; |
| save_to_json | Save results as a JSON file | `- save_path`: String, file path for saving; if it's a directory, the saved file name matches the input file name; <br>`- indent`: Integer, JSON formatting setting, default is 4; <br>`- ensure_ascii`: Boolean, JSON formatting setting, default is False; |
| save_to_img  | Save results as an image file | `- save_path`: String, file path for saving; if it's a directory, the saved file name matches the input file name; |

If you have obtained the configuration file, you can customize various settings of the facial recognition pipeline by simply modifying the `pipeline` parameter value in the `create_pipeline` method to the path of the pipeline configuration file.

For example, if your configuration file is saved at `./my_path/face_recognition.yaml`, you just need to execute:

```python
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="./my_path/face_recognition.yaml")
pipeline.build_index(data_root="face_demo_gallery", index_dir="face_gallery_index")
output = pipeline.predict("friends1.jpg")
for res in output:
    res.print()
    res.save_to_img("./output/")
```

#### 2.2.3 Adding and Deleting Operations in the Face Feature Library

If you wish to add more face images to the feature library, you can call the `add_index` method; to delete face image features, you can call the `delete_index` method.

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="face_recognition")

pipeline.add_index(data_root="add_gallery", index_dir="face_gallery_index")

pipeline.delete_index(data_root="delete_gallery", index_dir="face_gallery_index")
```

The `add_index` method parameters are described as follows:

| Parameter    | Description                                                                                                                                                                                                                    | Type  | Default |
|--------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------|---------|
| `data_root`  | The root directory of the dataset to be added. The data organization method is the same as when building the feature library. Refer to [Section 2.3 Data Organization for Feature Library Construction](###2.3-Data-Organization-for-Feature-Library-Construction). | `str` | None    |
| `index_dir`  | The save path of the feature library to which features are added. After successfully calling the `add_index` method, the face image features in `data_root` will be added to the face feature library originally saved at `index_dir`.                                 | `str` | None    |

The `delete_index` method parameters are described as follows:

| Parameter    | Description                                                                                                                                                                                                                         | Type  | Default |
|--------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------|---------|
| `data_root`  | The root directory of the dataset to be deleted. The data organization method is the same as when building the feature library. Refer to [Section 2.3 Data Organization for Feature Library Construction](#2.3-Data-Organization-for-Feature-Library-Construction). | `str` | None    |
| `index_dir`  | The save path of the feature library from which features are deleted. After successfully calling the `delete_index` method, the face image features in `data_root` will be deleted from the face feature library originally saved at `index_dir`.                             | `str` | None    |

### 2.3 Data Organization for Feature Library Construction

The face recognition pipeline example in PaddleX requires a pre-constructed feature library for face feature retrieval. If you wish to build a face feature library with private data, you need to organize the data as follows:

```bash
data_root             # Root directory of the dataset, the directory name can be changed
├── images            # Directory for saving images, the directory name can be changed
│   ├── ID0           # Identity ID name, preferably meaningful, such as a person's name
│   │   ├── xxx.jpg   # Image, nested directories are supported
│   │   ├── xxx.jpg   # Image, nested directories are supported
│   │       ...  
│   ├── ID1           # Identity ID name, preferably meaningful, such as a person's name
│   │   ...
└── gallery.txt       # Annotation file for the feature library dataset, the file name cannot be changed. Each line gives the path of the face image to be retrieved and the image feature label, separated by a space. Example content: images/Chandler/Chandler00037.jpg Chandler
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

    Obtain OCR results for an image.

    `POST /ocr`

    - The attributes of the request body are as follows:

        | Name | Type | Meaning | Required |
        |-|-|-|-|
        |`image`|`string`|The URL of an accessible image file or the Base64 encoded result of the image file content. |Yes|
        |`inferenceParams`|`object`|Inference parameters. |No|

        The attributes of```markdown
<details>
<summary>Python</summary>

```python
import base64
import requests

API_URL = "http://localhost:8080/ocr" # Service URL
image_path = "./demo.jpg"
output_image_path = "./out.jpg"

# Encode the local image to Base64
with open(image_path, "rb") as file:
    image_bytes = file.read()
    image_data = base64.b64encode(image_bytes).decode("ascii")

payload = {"image": image_data}  # Base64 encoded file content or image URL

# Call the API
response = requests.post(API_URL, json=payload)

# Process the response data
assert response.status_code == 200
result = response.json()["result"]
with open(output_image_path, "wb") as file:
    file.write(base64.b64decode(result["image"]))
print(f"Output image saved at {output_image_path}")
print("\nDetected texts:")
print(result["texts"])
```

</details>

<details>
<summary>C++</summary>

```cpp
#include <iostream>
#include "cpp-httplib/httplib.h" // https://github.com/Huiyicc/cpp-httplib
#include "nlohmann/json.hpp" // https://github.com/nlohmann/json
#include "base64.hpp" // https://github.com/tobiaslocker/base64

int main() {
    httplib::Client client("localhost:8080");
    const std::string imagePath = "./demo.jpg";
    const std::string outputImagePath = "./out.jpg";

    httplib::Headers headers = {
        {"Content-Type", "application/json"}
    };

    // Encode the local image to Base64
    std::ifstream file(imagePath, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        std::cerr << "Error reading file." << std::endl;
        return 1;
    }
    std::string bufferStr(reinterpret_cast<const char*>(buffer.data()), buffer.size());
    std::string encodedImage = base64::to_base64(bufferStr);

    nlohmann::json jsonObj;
    jsonObj["image"] = encodedImage;
    std::string body = jsonObj.dump();

    // Call the API
    auto response = client.Post("/ocr", headers, body, "application/json");
    // Process the response data
    if (response && response->status == 200) {
        nlohmann::json jsonResponse = nlohmann::json::parse(response->body);
        auto result = jsonResponse["result"];

        encodedImage = result["image"];
        std::string decodedString = base64::from_base64(encodedImage);
        std::vector<unsigned char> decodedImage(decodedString.begin(), decodedString.end());
        std::ofstream outputImage(outputImagePath, std::ios::binary | std::ios::out);
        if (outputImage.is_open()) {
            outputImage.write(reinterpret_cast<char*>(decodedImage.data()), decodedImage.size());
            outputImage.close();
            std::cout << "Output image saved at " << outputImagePath << std::endl;
        } else {
            std::cerr << "Unable to open file for writing: " << outputImagePath << std::endl;
        }

        auto texts = result["texts"];
        std::cout << "\nDetected texts:" << std::endl;
        for (const auto& text : texts) {
            std::cout << text << std::endl;
        }
    } else {
        std::cout << "Failed to send HTTP request." << std::endl;
        return 1;
    }

    return 0;
}

```

</details>
``````markdown
# Tutorial on Artificial Intelligence and Computer Vision

This tutorial, intended for numerous developers, covers the basics and applications of AI and Computer Vision.

<details>
<summary>Java</summary>

```java
import okhttp3.*;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Base64;

public class Main {
    public static void main(String[] args) throws IOException {
        String API_URL = "http://localhost:8080/ocr"; // Service URL
        String imagePath = "./demo.jpg"; // Local image path
        String outputImagePath = "./out.jpg"; // Output image path

        // Encode the local image to Base64
        File file = new File(imagePath);
        byte[] fileContent = java.nio.file.Files.readAllBytes(file.toPath());
        String imageData = Base64.getEncoder().encodeToString(fileContent);

        ObjectMapper objectMapper = new ObjectMapper();
        ObjectNode params = objectMapper.createObjectNode();
        params.put("image", imageData); // Base64-encoded file content or image URL

        // Create an OkHttpClient instance
        OkHttpClient client = new OkHttpClient();
        MediaType JSON = MediaType.get("application/json; charset=utf-8");
        RequestBody body = RequestBody.create(params.toString(), JSON);
        Request request = new Request.Builder()
                .url(API_URL)
                .post(body)
                .build();

        // Call the API and process the response
        try (Response response = client.newCall(request).execute()) {
            if (response.isSuccessful()) {
                String responseBody = response.body().string();
                JsonNode resultNode = objectMapper.readTree(responseBody);
                JsonNode result = resultNode.get("result");
                String base64Image = result.get("image").asText();
                JsonNode texts = result.get("texts");

                byte[] imageBytes = Base64.getDecoder().decode(base64Image);
                try (FileOutputStream fos = new FileOutputStream(outputImagePath)) {
                    fos.write(imageBytes);
                }
                System.out.println("Output image saved at " + outputImagePath);
                System.out.println("\nDetected texts: " + texts.toString());
            } else {
                System.err.println("Request failed with code: " + response.code());
            }
        }
    }
}
```

</details>

<details>
<summary>Go</summary>

```go
package main

import (
    "bytes"
    "encoding/base64"
    "encoding/json"
    "fmt"
    "io/ioutil"
    "net/http"
)

func main() {
    API_URL := "http://localhost:8080/ocr"
    imagePath := "./demo.jpg"
    outputImagePath := "./out.jpg"

    // Encode the local image to Base64
    imageBytes, err := ioutil.ReadFile(imagePath)
    if err != nil {
        fmt.Println("Error reading image file:", err)
        return
    }
    imageData := base64.StdEncoding.EncodeToString(imageBytes)

    payload := map[string]string{"image": imageData} // Base64-encoded file content or image URL
    payloadBytes, err := json.Marshal(payload)
    if err != nil {
        fmt.Println("Error marshaling payload:", err)
        return
    }

    // Call the API
    client := &http.Client{}
    req, err := http.NewRequest("POST", API_URL, bytes.NewBuffer(payloadBytes))
    if err != nil {
        fmt.Println("Error creating request:", err)
        return
    }

    res, err := client.Do(req)
    if err != nil {
        fmt.Println("Error sending request:", err)
        return
    }
    defer res.Body.Close()

    // Process the response
    body, err := ioutil.ReadAll(res.Body)
    if err != nil {
        fmt.Println("Error reading response body:", err)
        return
    }```markdown
# An English Tutorial on Artificial Intelligence and Computer Vision

This tutorial document is intended for numerous developers and covers content related to artificial intelligence and computer vision.

<details>
<summary>C#</summary>

```csharp
using System;
using System.IO;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json.Linq;

class Program
{
    static readonly string API_URL = "http://localhost:8080/ocr";
    static readonly string imagePath = "./demo.jpg";
    static readonly string outputImagePath = "./out.jpg";

    static async Task Main(string[] args)
    {
        var httpClient = new HttpClient();

        // Encode the local image to Base64
        byte[] imageBytes = File.ReadAllBytes(imagePath);
        string image_data = Convert.ToBase64String(imageBytes);

        var payload = new JObject{ { "image", image_data } }; // Base64 encoded file content or image URL
        var content = new StringContent(payload.ToString(), Encoding.UTF8, "application/json");

        // Call the API
        HttpResponseMessage response = await httpClient.PostAsync(API_URL, content);
        response.EnsureSuccessStatusCode();

        // Process the API response
        string responseBody = await response.Content.ReadAsStringAsync();
        JObject jsonResponse = JObject.Parse(responseBody);

        string base64Image = jsonResponse["result"]["image"].ToString();
        byte[] outputImageBytes = Convert.FromBase64String(base64Image);

        File.WriteAllBytes(outputImagePath, outputImageBytes);
        Console.WriteLine($"Output image saved at {outputImagePath}");
        Console.WriteLine("\nDetected texts:");
        Console.WriteLine(jsonResponse["result"]["texts"].ToString());
    }
}
```

</details>

<details>
<summary>Node.js</summary>

```js
const axios = require('axios');
const fs = require('fs');

const API_URL = 'http://localhost:8080/ocr';
const imagePath = './demo.jpg';
const outputImagePath = "./out.jpg";

let config = {
   method: 'POST',
   maxBodyLength: Infinity,
   url: API_URL,
   data: JSON.stringify({
    'image': encodeImageToBase64(imagePath)  // Base64 encoded file content or image URL
  })
};

// Encode the local image to Base64
function encodeImageToBase64(filePath) {
  const bitmap = fs.readFileSync(filePath);
  return Buffer.from(bitmap).toString('base64');
}

// Call the API
axios.request(config)
.then((response) => {
    // Process the API response
    const result = response.data["result"];
    const imageBuffer = Buffer.from(result["image"], 'base64');
    fs.writeFile(outputImagePath, imageBuffer, (err) => {
      if (err) throw err;
      console.log(`Output image saved at ${outputImagePath}`);
    });
    console.log("\nDetected texts:");
    console.log(result["texts"]);
})
.catch((error) => {
  console.log(error);
});
```

</details>

<details>
<summary>PHP</summary>

```php
<?php

$API_URL = "http://localhost:8080/ocr"; // Service URL
$image_path = "./demo.jpg";
$output_image_path = "./out.jpg";

// Encode the local image to Base64
$image_data = base64_encode(file_get_contents($image_path));
$payload = array("image" => $image_data); // Base64 encoded file content or image URL

// Call the API
$ch = curl_init($API_URL);
curl_setopt($ch, CURLOPT_POST, true);
curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($payload));
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
$response = curl_exec($ch);
curl_close($ch);

// Process the API response
$result = json_decode($response, true)["result"];
file_put_contents($output
```

<details>
<details>
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
