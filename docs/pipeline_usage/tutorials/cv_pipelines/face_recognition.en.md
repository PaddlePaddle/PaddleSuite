---
comments: true
---

# Face Recognition Pipeline Tutorial

## 1. Introduction to the Face Recognition Pipeline
Face recognition is a crucial component in the field of computer vision, aiming to automatically identify individuals by analyzing and comparing facial features. This task involves not only detecting faces in images but also extracting and matching facial features to find corresponding identity information in a database. Face recognition is widely used in security authentication, surveillance systems, social media, smart devices, and other scenarios.

The face recognition pipeline is an end-to-end system dedicated to solving face detection and recognition tasks. It can quickly and accurately locate face regions in images, extract facial features, and retrieve and compare them with pre-established features in a feature database to confirm identity information.

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/face_recognition/01.png">

<b>The face recognition pipeline includes a face detection module and a face feature module</b>, with several models in each module. Which models to use can be selected based on the benchmark data below. <b>If you prioritize model accuracy, choose models with higher accuracy; if you prioritize inference speed, choose models with faster inference; if you prioritize model size, choose models with smaller storage requirements</b>.

<details><summary> 👉Model List Details</summary>

<p><b>Face Detection Module</b>:</p>
<table>
<thead>
<tr>
<th>Model</th>
<th>AP (%)<br>Easy/Medium/Hard</th>
<th>GPU Inference Time (ms)</th>
<th>CPU Inference Time</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>BlazeFace</td>
<td>77.7/73.4/49.5</td>
<td></td>
<td></td>
<td>0.447</td>
<td>A lightweight and efficient face detection model</td>
</tr>
<tr>
<td>BlazeFace-FPN-SSH</td>
<td>83.2/80.5/60.5</td>
<td></td>
<td></td>
<td>0.606</td>
<td>Improved BlazeFace with FPN and SSH structures</td>
</tr>
<tr>
<td>PicoDet_LCNet_x2_5_face</td>
<td>93.7/90.7/68.1</td>
<td></td>
<td></td>
<td>28.9</td>
<td>Face detection model based on PicoDet_LCNet_x2_5</td>
</tr>
<tr>
<td>PP-YOLOE_plus-S_face</td>
<td>93.9/91.8/79.8</td>
<td></td>
<td></td>
<td>26.5</td>
<td>Face detection model based on PP-YOLOE_plus-S</td>
</tr>
</tbody>
</table>
<p>Note: The above accuracy metrics are evaluated on the WIDER-FACE validation set with an input size of 640x640. All GPU inference times are based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speeds are based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</p>
<p><b>Face Recognition Module</b>:</p>
<table>
<thead>
<tr>
<th>Model</th>
<th>Output Feature Dimension</th>
<th>AP (%)<br>AgeDB-30/CFP-FP/LFW</th>
<th>GPU Inference Time (ms)</th>
<th>CPU Inference Time</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>MobileFaceNet</td>
<td>128</td>
<td>96.28/96.71/99.58</td>
<td></td>
<td></td>
<td>4.1</td>
<td>Face recognition model trained on MS1Mv3 based on MobileFaceNet</td>
</tr>
<tr>
<td>ResNet50_face</td>
<td>512</td>
<td>98.12/98.56/99.77</td>
<td></td>
<td></td>
<td>87.2</td>
<td>Face recognition model trained on MS1Mv3 based on ResNet50</td>
</tr>
</tbody>
</table>
<p>Note: The above accuracy metrics are Accuracy scores measured on the AgeDB-30, CFP-FP, and LFW datasets, respectively. All GPU inference times are based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speeds are based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</p></details>

## 2. Quick Start
The pre-trained model pipelines provided by PaddleX can be quickly experienced. You can experience the effects of the face recognition pipeline online or locally using command-line or Python.

### 2.1 Online Experience

Oneline Experience is not supported at the moment.

### 2.2 Local Experience
> ❗ Before using the facial recognition pipeline locally, please ensure that you have completed the installation of the PaddleX wheel package according to the [PaddleX Installation Guide](../../../installation/installation.md).

#### 2.2.1 Command Line Experience

Command line experience is not supported at the moment.
#### 2.2.2 Integration via Python Script

Please download the [test image](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/friends1.jpg) for testing. In the example of running this pipeline, you need to pre-build a facial feature library. You can refer to the following instructions to download the official demo data to be used for subsequent construction of the facial feature library. You can use the following command to download the demo dataset to a specified folder:

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

<table>
<thead>
<tr>
<th>Parameter</th>
<th>Description</th>
<th>Type</th>
<th>Default</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>pipeline</code></td>
<td>The name of the pipeline or the path to the pipeline configuration file. If it is the pipeline name, it must be a pipeline supported by PaddleX.</td>
<td><code>str</code></td>
<td>None</td>
</tr>
<tr>
<td><code>device</code></td>
<td>The device for pipeline model inference. Supports: "gpu", "cpu".</td>
<td><code>str</code></td>
<td>"gpu"</td>
</tr>
<tr>
<td><code>use_hpip</code></td>
<td>Whether to enable high-performance inference, only available when the pipeline supports high-performance inference.</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
</tbody>
</table>
(2) Call the `build_index` method of the face recognition pipeline object to build the facial feature library. The specific parameters are described as follows:

<table>
<thead>
<tr>
<th>Parameter</th>
<th>Description</th>
<th>Type</th>
<th>Default</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>data_root</code></td>
<td>The root directory of the dataset, with data organization referring to <a href="#2.3-Data-Organization-for-Building-a-Feature-Library">Section 2.3: Data Organization for Building a Feature Library</a></td>
<td><code>str</code></td>
<td>None</td>
</tr>
<tr>
<td><code>index_dir</code></td>
<td>The save path for the feature library. After successfully calling the <code>build_index</code> method, two files will be generated in this path:<br> <code>"id_map.pkl"</code> saves the mapping relationship between image IDs and image feature labels;<br> <code>"vector.index"</code> stores the feature vectors of each image.</td>
<td><code>str</code></td>
<td>None</td>
</tr>
</tbody>
</table>
(3) Call the `predict` method of the face recognition pipeline object for inference prediction: The `predict` method parameter is `x`, used to input data to be predicted, supporting multiple input methods, as shown in the following examples:

<table>
<thead>
<tr>
<th>Parameter Type</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>Python Var</td>
<td>Supports directly passing in Python variables, such as image data represented by <code>numpy.ndarray</code>.</td>
</tr>
<tr>
<td><code>str</code></td>
<td>Supports passing in the file path of the data to be predicted, such as the local path of an image file: <code>/root/data/img.jpg</code>.</td>
</tr>
<tr>
<td><code>str</code></td>
<td>Supports passing in the URL of the data file to be predicted, such as the network URL of an image file: <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_001.png">Example</a>.</td>
</tr>
<tr>
<td><code>str</code></td>
<td>Supports passing in a local directory containing the data files to be predicted, such as the local path: <code>/root/data/</code>.</td>
</tr>
<tr>
<td><code>dict</code></td>
<td>Supports passing in a dictionary type, where the key needs to correspond to the specific task, such as "img" for image classification tasks, and the value of the dictionary supports the above types of data, for example: <code>{"img": "/root/data1"}</code>.</td>
</tr>
<tr>
<td><code>list</code></td>
<td>Supports passing in a list, where the list elements need to be the above types of data, such as <code>[numpy.ndarray, numpy.ndarray], ["/root/data/img1.jpg", "/root/data/img2.jpg"], ["/root/data1", "/root/data2"], [{"img": "/root/data1"}, {"img": "/root/data2/img.jpg"}]</code>.</td>
</tr>
</tbody>
</table>
(4) Obtain the prediction results by calling the `predict` method: The `predict` method is a `generator`, so prediction results need to be obtained through iteration. The `predict` method predicts data in batches, so the prediction results are in the form of a list.

(5) Process the prediction results: The prediction result for each sample is of type `dict`, and it supports printing or saving to a file. The supported file types depend on the specific pipeline, such as:

<table>
<thead>
<tr>
<th>Method</th>
<th>Description</th>
<th>Method Parameters</th>
</tr>
</thead>
<tbody>
<tr>
<td>print</td>
<td>Print results to the terminal</td>
<td><code>- format_json</code>: Boolean, whether to format the output with JSON indentation, default is True; <br><code>- indent</code>: Integer, JSON formatting setting, effective only when format_json is True, default is 4; <br><code>- ensure_ascii</code>: Boolean, JSON formatting setting, effective only when format_json is True, default is False;</td>
</tr>
<tr>
<td>save_to_json</td>
<td>Save results as a JSON file</td>
<td><code>- save_path</code>: String, file path for saving; if it's a directory, the saved file name matches the input file name; <br><code>- indent</code>: Integer, JSON formatting setting, default is 4; <br><code>- ensure_ascii</code>: Boolean, JSON formatting setting, default is False;</td>
</tr>
<tr>
<td>save_to_img</td>
<td>Save results as an image file</td>
<td><code>- save_path</code>: String, file path for saving; if it's a directory, the saved file name matches the input file name;</td>
</tr>
</tbody>
</table>
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

<table>
<thead>
<tr>
<th>Parameter</th>
<th>Description</th>
<th>Type</th>
<th>Default</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>data_root</code></td>
<td>The root directory of the dataset to be added. The data organization method is the same as when building the feature library. Refer to <a href="###2.3-Data-Organization-for-Feature-Library-Construction">Section 2.3 Data Organization for Feature Library Construction</a>.</td>
<td><code>str</code></td>
<td>None</td>
</tr>
<tr>
<td><code>index_dir</code></td>
<td>The save path of the feature library to which features are added. After successfully calling the <code>add_index</code> method, the face image features in <code>data_root</code> will be added to the face feature library originally saved at <code>index_dir</code>.</td>
<td><code>str</code></td>
<td>None</td>
</tr>
</tbody>
</table>
The `delete_index` method parameters are described as follows:

<table>
<thead>
<tr>
<th>Parameter</th>
<th>Description</th>
<th>Type</th>
<th>Default</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>data_root</code></td>
<td>The root directory of the dataset to be deleted. The data organization method is the same as when building the feature library. Refer to <a href="#2.3-Data-Organization-for-Feature-Library-Construction">Section 2.3 Data Organization for Feature Library Construction</a>.</td>
<td><code>str</code></td>
<td>None</td>
</tr>
<tr>
<td><code>index_dir</code></td>
<td>The save path of the feature library from which features are deleted. After successfully calling the <code>delete_index</code> method, the face image features in <code>data_root</code> will be deleted from the face feature library originally saved at <code>index_dir</code>.</td>
<td><code>str</code></td>
<td>None</td>
</tr>
</tbody>
</table>
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

🚀 <b>High-Performance Inference</b>: In actual production environments, many applications have stringent standards for the performance metrics of deployment strategies (especially response speed) to ensure efficient system operation and smooth user experience. To this end, PaddleX provides high-performance inference plugins aimed at deeply optimizing model inference and pre/post-processing to significantly speed up the end-to-end process. For detailed high-performance inference procedures, please refer to the [PaddleX High-Performance Inference Guide](../../../pipeline_deploy/high_performance_inference.md).

☁️ <b>Service-Oriented Deployment</b>: Service-oriented deployment is a common deployment form in actual production environments. By encapsulating inference functionality as services, clients can access these services through network requests to obtain inference results. PaddleX supports users in achieving service-oriented deployment of pipelines at low cost. For detailed service-oriented deployment procedures, please refer to the [PaddleX Service-Oriented Deployment Guide](../../../pipeline_deploy/service_deploy.md).

Below are the API reference and multi-language service invocation examples:

<details><summary>API Reference</summary>

<p>For all operations provided by the service:</p>
<ul>
<li>The response body and the request body of POST requests are both JSON data (JSON objects).</li>
<li>When the request is successfully processed, the response status code is <code>200</code>, and the attributes of the response body are as follows:</li>
</ul>
<table>
<thead>
<tr>
<th>Name</th>
<th>Type</th>
<th>Meaning</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>errorCode</code></td>
<td><code>integer</code></td>
<td>Error code. Fixed to <code>0</code>.</td>
</tr>
<tr>
<td><code>errorMsg</code></td>
<td><code>string</code></td>
<td>Error description. Fixed to <code>"Success"</code>.</td>
</tr>
</tbody>
</table>
<p>The response body may also have a <code>result</code> attribute of type <code>object</code>, which stores the operation result information.</p>
<ul>
<li>When the request is not successfully processed, the attributes of the response body are as follows:</li>
</ul>
<table>
<thead>
<tr>
<th>Name</th>
<th>Type</th>
<th>Meaning</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>errorCode</code></td>
<td><code>integer</code></td>
<td>Error code. Same as the response status code.</td>
</tr>
<tr>
<td><code>errorMsg</code></td>
<td><code>string</code></td>
<td>Error description.</td>
</tr>
</tbody>
</table>
<p>The operations provided by the service are as follows:</p>
<ul>
<li><b><code>infer</code></b></li>
</ul>
<p>Obtain OCR results for an image.</p>
<p><code>POST /ocr</code></p>
<ul>
<li>The attributes of the request body are as follows:</li>
</ul>
<table>
<thead>
<tr>
<th>Name</th>
<th>Type</th>
<th>Meaning</th>
<th>Required</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>image</code></td>
<td><code>string</code></td>
<td>The URL of an accessible image file or the Base64 encoded result of the image file content.</td>
<td>Yes</td>
</tr>
<tr>
<td><code>inferenceParams</code></td>
<td><code>object</code></td>
<td>Inference parameters.</td>
<td>No</td>
</tr>
</tbody>
</table>
<p>The attributes of```markdown</p>
<details>
<summary>Python</summary>


<pre><code class="language-python">import base64
import requests

API_URL = &quot;http://localhost:8080/ocr&quot; # Service URL
image_path = &quot;./demo.jpg&quot;
output_image_path = &quot;./out.jpg&quot;

# Encode the local image to Base64
with open(image_path, &quot;rb&quot;) as file:
    image_bytes = file.read()
    image_data = base64.b64encode(image_bytes).decode(&quot;ascii&quot;)

payload = {&quot;image&quot;: image_data}  # Base64 encoded file content or image URL

# Call the API
response = requests.post(API_URL, json=payload)

# Process the response data
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

    // Encode the local image to Base64
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

    // Call the API
    auto response = client.Post(&quot;/ocr&quot;, headers, body, &quot;application/json&quot;);
    // Process the response data
    if (response &amp;&amp; response-&gt;status == 200) {
        nlohmann::json jsonResponse = nlohmann::json::parse(response-&gt;body);
        auto result = jsonResponse[&quot;result&quot;];

        encodedImage = result[&quot;image&quot;];
        std::string decodedString = base64::from_base64(encodedImage);
        std::vector&lt;unsigned char&gt; decodedImage(decodedString.begin(), decodedString.end());
        std::ofstream outputImage(outputImagePath, std::ios::binary | std::ios::out);
        if (outputImage.is_open()) {
            outputImage.write(reinterpret_cast&lt;char*&gt;(decodedImage.data()), decodedImage.size());
            outputImage.close();
            std::cout &lt;&lt; &quot;Output image saved at &quot; &lt;&lt; outputImagePath &lt;&lt; std::endl;
        } else {
            std::cerr &lt;&lt; &quot;Unable to open file for writing: &quot; &lt;&lt; outputImagePath &lt;&lt; std::endl;
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
``````markdown
# Tutorial on Artificial Intelligence and Computer Vision

This tutorial, intended for numerous developers, covers the basics and applications of AI and Computer Vision.

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
        String API_URL = &quot;http://localhost:8080/ocr&quot;; // Service URL
        String imagePath = &quot;./demo.jpg&quot;; // Local image path
        String outputImagePath = &quot;./out.jpg&quot;; // Output image path

        // Encode the local image to Base64
        File file = new File(imagePath);
        byte[] fileContent = java.nio.file.Files.readAllBytes(file.toPath());
        String imageData = Base64.getEncoder().encodeToString(fileContent);

        ObjectMapper objectMapper = new ObjectMapper();
        ObjectNode params = objectMapper.createObjectNode();
        params.put(&quot;image&quot;, imageData); // Base64-encoded file content or image URL

        // Create an OkHttpClient instance
        OkHttpClient client = new OkHttpClient();
        MediaType JSON = MediaType.get(&quot;application/json; charset=utf-8&quot;);
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

    // Encode the local image to Base64
    imageBytes, err := ioutil.ReadFile(imagePath)
    if err != nil {
        fmt.Println(&quot;Error reading image file:&quot;, err)
        return
    }
    imageData := base64.StdEncoding.EncodeToString(imageBytes)

    payload := map[string]string{&quot;image&quot;: imageData} // Base64-encoded file content or image URL
    payloadBytes, err := json.Marshal(payload)
    if err != nil {
        fmt.Println(&quot;Error marshaling payload:&quot;, err)
        return
    }

    // Call the API
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

    // Process the response
    body, err := ioutil.ReadAll(res.Body)
    if err != nil {
        fmt.Println(&quot;Error reading response body:&quot;, err)
        return
    }```markdown
# An English Tutorial on Artificial Intelligence and Computer Vision

This tutorial document is intended for numerous developers and covers content related to artificial intelligence and computer vision.

&lt;details&gt;
&lt;summary&gt;C#&lt;/summary&gt;

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
static readonly string API_URL = &quot;http://localhost:8080/ocr&quot;;
static readonly string imagePath = &quot;./demo.jpg&quot;;
static readonly string outputImagePath = &quot;./out.jpg&quot;;

static async Task Main(string[] args)
{
var httpClient = new HttpClient();

// Encode the local image to Base64
byte[] imageBytes = File.ReadAllBytes(imagePath);
string image_data = Convert.ToBase64String(imageBytes);

var payload = new JObject{ { &quot;image&quot;, image_data } }; // Base64 encoded file content or image URL
var content = new StringContent(payload.ToString(), Encoding.UTF8, &quot;application/json&quot;);

// Call the API
HttpResponseMessage response = await httpClient.PostAsync(API_URL, content);
response.EnsureSuccessStatusCode();

// Process the API response
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

const API_URL = 'http://localhost:8080/ocr';
const imagePath = './demo.jpg';
const outputImagePath = &quot;./out.jpg&quot;;

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
.then((response) =&gt; {
    // Process the API response
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

📱 <b>Edge Deployment</b>: Edge deployment is a method where computing and data processing functions are placed on the user's device itself, allowing the device to process data directly without relying on remote servers. PaddleX supports deploying models on edge devices such as Android. For detailed edge deployment procedures, please refer to the [PaddleX Edge Deployment Guide](../../../pipeline_deploy/edge_deploy.en.md).
You can choose an appropriate method to deploy your model pipeline based on your needs, and proceed with subsequent AI application integration.


## 4. Custom Development
If the default model weights provided by the Face Recognition Pipeline do not meet your expectations in terms of accuracy or speed for your specific scenario, you can try to further <b>fine-tune</b> the existing models using <b>your own domain-specific or application-specific data</b> to enhance the recognition performance of the pipeline in your scenario.

### 4.1 Model Fine-tuning
Since the Face Recognition Pipeline consists of two modules (face detection and face recognition), the suboptimal performance of the pipeline may stem from either module.

You can analyze images with poor recognition results. If you find that many faces are not detected during the analysis, it may indicate deficiencies in the face detection model. In this case, you need to refer to the [Custom Development](../../../module_usage/tutorials/cv_modules/face_detection.en.md#IV.-Custom-Development) section in the [Face Detection Module Development Tutorial](../../../module_usage/tutorials/cv_modules/face_detection.en.md) and use your private dataset to fine-tune the face detection model. If matching errors occur in detected faces, it suggests that the face feature model needs further improvement. You should refer to the [Custom Development](../../../module_usage/tutorials/cv_modules/face_feature.en.md#IV.-Custom-Development) section in the [Face Feature Module Development Tutorial](../../../module_usage/tutorials/cv_modules/face_feature.en.md) to fine-tune the face feature model.

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
PaddleX supports various mainstream hardware devices such as NVIDIA GPUs, Kunlun XPU, Ascend NPU, and Cambricon MLU. <b>Simply modifying the `--device` parameter</b> allows seamless switching between different hardware.

For example, when running the face recognition pipeline using Python and changing the running device from an NVIDIA GPU to an Ascend NPU, you only need to modify the `device` in the script to `npu`:

```python
from paddlex import create_pipeline

pipeline = create_pipeline(
    pipeline="face_recognition",
    device="npu:0" # gpu:0 --> npu:0
)
```
If you want to use the face recognition pipeline on more types of hardware, please refer to the [PaddleX Multi-device Usage Guide](../../../other_devices_support/multi_devices_use_guide.en.md).
