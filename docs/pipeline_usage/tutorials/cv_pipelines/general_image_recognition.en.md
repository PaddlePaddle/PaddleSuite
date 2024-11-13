---
comments: true
---

# General Image Recognition Pipeline Usage Tutorial

## 1. Introduction to the General Image Recognition Pipeline

The General Image Recognition Pipeline aims to solve the problem of open-domain object localization and recognition. Currently, PaddleX's General Image Recognition Pipeline supports PP-ShiTuV2.

PP-ShiTuV2 is a practical general image recognition system mainly composed of three modules: mainbody detection module, image feature module, and vector retrieval module. The system integrates and improves various strategies in multiple aspects, including backbone network, loss function, data augmentation, learning rate scheduling, regularization, pre-trained model, and model pruning and quantization. It optimizes each module and ultimately achieves better performance in multiple application scenarios.

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/pipelines/general_image_recognition/pp_shitu_v2.jpg">

<b>The General Image Recognition Pipeline includes the mainbody detection module and the image feature module</b>, with several models to choose. You can select the model to use based on the benchmark data below. <b>If you prioritize model precision, choose a model with higher precision. If you prioritize inference speed, choose a model with faster inference. If you prioritize model storage size, choose a model with a smaller storage size</b>.

<summary> 👉Model List Details</summary>

<b>Object Detection Module:</b>

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
    <td>PP-ShiTuV2_det</td>
    <td>41.5</td>
    <td>62.0</td>
    <td>33.7</td>
    <td>537.0</td>
    <td>27.54</td>
    <td>An mainbody detection model based on PicoDet_LCNet_x2_5, which may detect multiple common objects simultaneously.</td>
  </tr>
</table>

Note: The above accuracy metrics are based on the private mainbody detection dataset.

<b>Image Feature Module:</b>

<table>
  <tr>
    <th>Model</th>
    <th>Recall@1 (%)</th>
    <th>GPU Inference Time (ms)</th>
    <th>CPU Inference Time (ms)</th>
    <th>Model Size (M)</th>
    <th>Description</th>
  </tr>
  <tr>
    <td>PP-ShiTuV2_rec</td>
    <td>84.2</td>
    <td>5.23428</td>
    <td>19.6005</td>
    <td>16.3 M</td>
    <td rowspan="3">PP-ShiTuV2 is a general image feature system consisting of three modules: mainbody detection, feature extraction, and vector retrieval. These models are part of the feature extraction module, and different models can be selected based on system requirements.</td>
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

Note: The above accuracy metrics are based on AliProducts Recall@1. All GPU inference times are based on NVIDIA Tesla T4 machines with FP32 precision. CPU inference speeds are based on Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.

## 2. Quick Start

The pre-trained model pipelines provided by PaddleX can be quickly experienced. You can use Python to experience locally.

### 2.1 Online Experience

Not supported yet.

### 2.2 Local Experience

> ❗ Before using the General Image Recognition Pipeline locally, please ensure you have installed the PaddleX wheel package according to the [PaddleX Installation Tutorial](../../../installation/installation.en.md).

#### 2.2.1 Command Line Experience

The pipeline does not support command line experience at this time.

By default, the built-in General Image Recognition Pipeline configuration file is used. If you want to change it, you can run the following command to obtain:

<details><summary> 👉Click to Expand</summary>

<pre><code class="language-bash">paddlex --get_pipeline_config PP-ShiTuV2
</code></pre>
<p>After execution, the General Image Recognition Pipeline configuration file will be saved in the current directory. If you want to customize the save location, you can run the following command (assuming the custom save location is <code>./my_path</code>):</p>
<pre><code class="language-bash">paddlex --get_pipeline_config PP-ShiTuV2 --save_path ./my_path
</code></pre></details>

#### 2.2.2 Python Script Integration

* In the example of using this pipeline, a feature vector library needs to be built beforehand. You can download the officially provided drink recognition test dataset [drink_dataset_v2.0](https://paddle-model-ecology.bj.bcebos.com/paddlex/data/drink_dataset_v2.0.tar) to build the feature vector library. If you want to use a private dataset, you can refer to [Section 2.3 Data Organization for Building the Feature Library](#23-data-organization-for-building-the-feature-library). After that, you can quickly build the feature vector library and predict using the General Image Recognition Pipeline with just a few lines of code.

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="PP-ShiTuV2")

pipeline.build_index(data_root="drink_dataset_v2.0/", index_dir="index_dir")

output = pipeline.predict("./drink_dataset_v2.0/test_images/", index_dir="index_dir")
for res in output:
    res.print()
    res.save_to_img("./output/")

````

In the above Python script, the following steps are executed:

(1) Call the `create_pipeline` function to create a general image recognition pipeline object. The specific parameter descriptions are as follows:

<table>
<thead>
<tr>
<th>Parameter</th>
<th>Parameter Description</th>
<th>Parameter Type</th>
<th>Default Value</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>pipeline</code></td>
<td>The name of the pipeline or the path to the pipeline configuration file. If it is the name of the pipeline, it must be a pipeline supported by PaddleX.</td>
<td><code>str</code></td>
<td>None</td>
</tr>
<tr>
<td><code>index_dir</code></td>
<td>The directory where the retrieval database files used for pipeline inference are located. If this parameter is not passed, <code>index_dir</code> needs to be specified in <code>predict()</code>.</td>
<td><code>str</code></td>
<td>None</td>
</tr>
<tr>
<td><code>device</code></td>
<td>The inference device for the pipeline model. Supports: "gpu", "cpu".</td>
<td><code>str</code></td>
<td><code>gpu</code></td>
</tr>
<tr>
<td><code>use_hpip</code></td>
<td>Whether to enable high-performance inference, which is only available when the pipeline supports it.</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
</tbody>
</table>

(2) Call the `build_index` function of the general image recognition pipeline object to build the feature vector library. The specific parameters are described as follows:

<table>
<thead>
<tr>
<th>Parameter</th>
<th>Parameter Description</th>
<th>Parameter Type</th>
<th>Default Value</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>data_root</code></td>
<td>The root directory of the dataset. The data organization method refers to <a href="#2.3-Data-Organization-for-Building-the-Feature-Library">Section 2.3 Data Organization for Building the Feature Library</a></td>
<td><code>str</code></td>
<td>None</td>
</tr>
<tr>
<td><code>index_dir</code></td>
<td>The save path for the feature library. After successfully calling the <code>build_index</code> function, two files will be generated in this path: <code>"id_map.pkl"</code> saves the mapping relationship between image IDs and image feature labels; <code>"vector.index"</code> stores the feature vectors of each image.</td>
<td><code>str</code></td>
<td>None</td>
</tr>
</tbody>
</table>

(3) Call the `predict` function of the general image recognition pipeline object for inference prediction: The `predict` function parameter is `input`, which is used to input the data to be predicted, supporting multiple input methods. Specific examples are as follows:

<table>
<thead>
<tr>
<th>Parameter Type</th>
<th>Parameter Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>Python Var</td>
<td>Supports directly passing in Python variables, such as <code>numpy.ndarray</code> representing image data.</td>
</tr>
<tr>
<td>str</td>
<td>Supports passing in the file path of the data to be predicted, such as the local path of an image file: <code>/root/data/img.jpg</code>.</td>
</tr>
<tr>
<td>str</td>
<td>Supports passing in the URL of the data file to be predicted, such as the network URL of an image file: <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/yuanqisenlin.jpeg">Example</a>.</td>
</tr>
<tr>
<td>str</td>
<td>Supports passing in a local directory that contains the data files to be predicted, such as the local path: <code>/root/data/</code>.</td>
</tr>
<tr>
<td>dict</td>
<td>Supports passing in a dictionary type, where the key needs to correspond to the specific task, such as "img" for image classification tasks. The value of the dictionary supports the above types of data, for example: <code>{"img": "/root/data1"}</code>.</td>
</tr>
<tr>
<td>list</td>
<td>Supports passing in a list, where the elements of the list need to be the above types of data, such as <code>[numpy.ndarray, numpy.ndarray], ["/root/data/img1.jpg", "/root/data/img2.jpg"]</code>, <code>["/root/data1", "/root/data2"]</code>, <code>[{"img": "/root/data1"}, {"img": "/root/data2/img.jpg"}]</code>.</td>
</tr>
</tbody>
</table>

Additionally, the `predict` method supports the `index_dir` parameter for setting the retrieval database:

<table>
<thead>
<tr>
<th>Parameter Type</th>
<th>Parameter Description</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>index_dir</code></td>
<td>The directory where the retrieval database files used for pipeline inference are located. If this parameter is not passed, the default retrieval database specified through the <code>index_dir</code> parameter in <code>create_pipeline()</code> will be used.</td>
</tr>
</tbody>
</table>

(4) Obtain the prediction results by calling the `predict` method: The `predict` method is a `generator`, so prediction results need to be obtained by iteration. The `predict` method predicts data in batches.

(5) Process the prediction results: The prediction result for each sample is of `dict` type and supports printing or saving to a file. The supported save types are related to the specific pipeline, such as:

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
<td>Print the results to the terminal</td>
<td><code>- format_json</code>: bool type, whether to use json indentation formatting for the output content, default is True;<br><code>- indent</code>: int type, json formatting setting, only effective when format_json is True, default is 4;<br><code>- ensure_ascii</code>: bool type, json formatting setting, only effective when format_json is True, default is False;</td>
</tr>
<tr>
<td>save_to_json</td>
<td>Save the results as a json-formatted file</td>
<td><code>- save_path</code>: str type, the save file path. When it is a directory, the saved file naming is consistent with the input file type naming;<br><code>- indent</code>: int type, json formatting setting, default is 4;<br><code>- ensure_ascii</code>: bool type, json formatting setting, default is False;</td>
</tr>
<tr>
<td>save_to_img</td>
<td>Save the results as an image-formatted file</td>
<td><code>- save_path</code>: str type, the save file path. When it is a directory, the saved file naming is consistent with the input file type naming;</td>
</tr>
</tbody>
</table>

If you have a configuration file, you can customize the configurations for the general image recognition pipeline by modifying the `pipeline` parameter value in the `create_pipeline` method to the path of the pipeline configuration file.

For example, if your configuration file is saved at `./my_path/PP-ShiTuV2.yaml`, you only need to execute:

```python
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="./my_path/PP-ShiTuV2.yaml", index_dir="index_dir")

output = pipeline.predict("./drink_dataset_v2.0/test_images/")
for res in output:
    res.print()
    res.save_to_img("./output/")
```

#### 2.2.3 Add or Remove Features from the Feature Library

If you want to add more images to the feature library, you can call the `append_index` function; to remove image features, you can call the `remove_index` function.

```python
from paddlex import create_pipeline

pipeline = create_pipeline("PP-ShiTuV2")
pipeline.build_index(data_root="drink_dataset_v2.0/", index_dir="index_dir", index_type="IVF")
pipeline.append_index(data_root="drink_dataset_v2.0/", index_dir="index_dir", index_type="IVF")
pipeline.remove_index(data_root="drink_dataset_v2.0/", index_dir="index_dir", index_type="IVF")
```

The parameter descriptions for the above methods are as follows:

<table>
<thead>
<tr>
<th>Parameter</th>
<th>Description</th>
<th>Type</th>
<th>Default Value</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>data_root</code></td>
<td>The root directory of the dataset to be added. The data organization should be the same as when building the feature library, refer to <a href="#2.3-Data-Organization-for-Building-the-Feature-Library">Section 2.3 Data Organization for Building the Feature Library</a></td>
<td><code>str</code></td>
<td>None</td>
</tr>
<tr>
<td><code>index_dir</code></td>
<td>The storage directory for the feature library. In <code>append_index</code> and <code>remove_index</code>, it is also the path of the feature library to be modified (or deleted).</td>
<td><code>str</code></td>
<td>None</td>
</tr>
<tr>
<td><code>index_type</code></td>
<td>Supports <code>HNSW32</code>, <code>IVF</code>, <code>Flat</code>. Among them, <code>HNSW32</code> has faster retrieval speed and higher accuracy but does not support the <code>remove_index()</code> operation; <code>IVF</code> has faster retrieval speed but relatively lower accuracy, and supports <code>append_index()</code> and <code>remove_index()</code> operations; <code>Flat</code> has lower retrieval speed but higher accuracy, and supports <code>append_index()</code> and <code>remove_index()</code> operations.</td>
<td><code>str</code></td>
<td><code>HNSW32</code></td>
</tr>
<tr>
<td><code>metric_type</code></td>
<td>Supports: <code>IP</code>, Inner Product; <code>L2</code>, Euclidean Distance.</td>
<td><code>str</code></td>
<td><code>IP</code></td>
</tr>
</tbody>
</table>

### 2.3 Data Organization for Building the Feature Library

The PaddleX general image recognition pipeline requires a pre-built feature library for feature retrieval. If you want to build a feature vector library with private data, you need to organize the data as follows:

```bash
data_root             # Root directory of the dataset, the directory name can be changed
├── images            # Directory for saving images, the directory name can be changed
│   │   ...
└── gallery.txt       # Annotation file for the feature library dataset, the file name cannot be changed. Each line gives the path of the image to be retrieved and the image label, separated by a space, for example: “0/0.jpg label”
```

## 3. Development Integration/Deployment
If the pipeline meets your requirements for inference speed and accuracy, you can proceed directly with development integration/deployment.

If you need to apply the pipeline directly in your Python project, refer to the example code in [2.2.2 Python Script Integration](#222-python-script-integration).

Additionally, PaddleX provides three other deployment methods, detailed as follows:

🚀 <b>High-Performance Inference</b>: In actual production environments, many applications have stringent standards for the performance metrics of deployment strategies (especially response speed) to ensure efficient system operation and smooth user experience. To this end, PaddleX provides high-performance inference plugins aimed at deeply optimizing model inference and pre/post-processing for significant end-to-end speedups. For detailed high-performance inference procedures, refer to the [PaddleX High-Performance Inference Guide](../../../pipeline_deploy/high_performance_inference.en.md).

☁️ <b>Service-Oriented Deployment</b>: Service-oriented deployment is a common deployment form in actual production environments. By encapsulating inference functions as services, clients can access these services through network requests to obtain inference results. PaddleX supports users in achieving low-cost service-oriented deployment of pipelines. For detailed service-oriented deployment procedures, refer to the [PaddleX Service-Oriented Deployment Guide](../../../pipeline_deploy/service_deploy.en.md).

Below are the API references and multi-language service invocation examples:

<details><summary>API Reference</summary>

<p>For all operations provided by the service:</p>
<ul>
<li>Both the response body and the request body for POST requests are JSON data (JSON objects).</li>
<li>When the request is processed successfully, the response status code is <code>200</code>, and the response body properties are as follows:</li>
</ul>
<table>
<thead>
<tr>
<th>Name</th>
<th>Type</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>errorCode</code></td>
<td><code>integer</code></td>
<td>Error code. Fixed as <code>0</code>.</td>
</tr>
<tr>
<td><code>errorMsg</code></td>
<td><code>string</code></td>
<td>Error message. Fixed as <code>"Success"</code>.</td>
</tr>
</tbody>
</table>
<p>The response body may also have a <code>result</code> property of type <code>object</code>, which stores the operation result information.</p>
<ul>
<li>When the request is not processed successfully, the response body properties are as follows:</li>
</ul>
<table>
<thead>
<tr>
<th>Name</th>
<th>Type</th>
<th>Description</th>
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
<td>Error message.</td>
</tr>
</tbody>
</table>
<p>Operations provided by the service are as follows:</p>
<ul>
<li><b><code>infer</code></b></li>
</ul>
<p>Classify images.</p>
<p><code>POST /image-classification</code></p>
<ul>
<li>The request body properties are as follows:</li>
</ul>
<table>
<thead>
<tr>
<th>Name</th>
<th>Type</th>
<th>Description</th>
<th>Required</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>image</code></td>
<td><code>string</code></td>
<td>The URL of an image file accessible by the service or the Base64 encoded result of the image file content.</td>
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
<p>The properties of <code>inferenceParams</code> are as follows:</p>
<table>
<thead>
<tr>
<th>Name</th>
<th>Type</th>
<th>Description</th>
<th>Required</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>topK</code></td>
<td><code>integer</code></td>
<td>Only the top <code>topK</code> categories with the highest scores will be retained in the results.</td>
<td>No</td>
</tr>
</tbody>
</table>
<ul>
<li>When the request is processed successfully, the <code>result</code> of the response body has the following properties:</li>
</ul>
<table>
<thead>
<tr>
<th>Name</th>
<th>Type</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>categories</code></td>
<td><code>array</code></td>
<td>Image category information.</td>
</tr>
<tr>
<td><code>image</code></td>
<td><code>string</code></td>
<td>The image classification result image. The image is in JPEG format and encoded using Base64.</td>
</tr>
</tbody>
</table>
<p>Each element in <code>categories</code> is an <code>object</code> with the following properties:</p>
<table>
<thead>
<tr>
<th>Name</th>
<th>Type</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>id</code></td>
<td><code>integer</code></td>
<td>Category ID.</td>
</tr>
<tr>
<td><code>name</code></td>
<td><code>string</code></td>
<td>Category name.</td>
</tr>
<tr>
<td><code>score</code></td>
<td><code>number</code></td>
<td>Category score.</td>
</tr>
</tbody>
</table>
<p>An example of <code>result</code> is as follows:</p>
<pre><code class="language-json">{
&quot;categories&quot;: [
{
&quot;id&quot;: 5,
&quot;name&quot;: &quot;Rabbit&quot;,
&quot;score&quot;: 0.93
}
],
&quot;image&quot;: &quot;xxxxxx&quot;
}
</code></pre></details>

<details><summary>Multi-Language Service Invocation Examples</summary>

<details>
<summary>Python</summary>


<pre><code class="language-python">import base64
import requests

API_URL = &quot;http://localhost:8080/image-classification&quot;
image_path = &quot;./demo.jpg&quot;
output_image_path = &quot;./out.jpg&quot;

with open(image_path, &quot;rb&quot;) as file:
    image_bytes = file.read()
    image_data = base64.b64encode(image_bytes).decode(&quot;ascii&quot;)

payload = {&quot;image&quot;: image_data}

response = requests.post(API_URL, json=payload)

assert response.status_code == 200
result = response.json()[&quot;result&quot;]
with open(output_image_path, &quot;wb&quot;) as file:
    file.write(base64.b64decode(result[&quot;image&quot;]))
print(f&quot;Output image saved at {output_image_path}&quot;)
print(&quot;\nCategories:&quot;)
print(result[&quot;categories&quot;])
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

    auto response = client.Post(&quot;/image-classification&quot;, headers, body, &quot;application/json&quot;);
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

        auto categories = result[&quot;categories&quot;];
        std::cout &lt;&lt; &quot;\nCategories:&quot; &lt;&lt; std::endl;
        for (const auto&amp; category : categories) {
            std::cout &lt;&lt; category &lt;&lt; std::endl;
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
        String API_URL = &quot;http://localhost:8080/image-classification&quot;;
        String imagePath = &quot;./demo.jpg&quot;;
        String outputImagePath = &quot;./out.jpg&quot;;

        File file = new File(imagePath);
        byte[] fileContent = java.nio.file.Files.readAllBytes(file.toPath());
        String imageData = Base64.getEncoder().encodeToString(fileContent);

        ObjectMapper objectMapper = new ObjectMapper();
        ObjectNode params = objectMapper.createObjectNode();
        params.put(&quot;image&quot;, imageData);

        OkHttpClient client = new OkHttpClient();
        MediaType JSON = MediaType.Companion.get(&quot;application/json; charset=utf-8&quot;);
        RequestBody body = RequestBody.Companion.create(params.toString(), JSON);
        Request request = new Request.Builder()
                .url(API_URL)
                .post(body)
                .build();

        try (Response response = client.newCall(request).execute()) {
            if (response.isSuccessful()) {
                String responseBody = response.body().string();
                JsonNode resultNode = objectMapper.readTree(responseBody);
                JsonNode result = resultNode.get(&quot;result&quot;);
                String base64Image = result.get(&quot;image&quot;).asText();
                JsonNode categories = result.get(&quot;categories&quot;);

                byte[] imageBytes = Base64.getDecoder().decode(base64Image);
                try (FileOutputStream fos = new FileOutputStream(outputImagePath)) {
                    fos.write(imageBytes);
                }
                System.out.println(&quot;Output image saved at &quot; + outputImagePath);
                System.out.println(&quot;\nCategories: &quot; + categories.toString());
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
    API_URL := &quot;http://localhost:8080/image-classification&quot;
    imagePath := &quot;./demo.jpg&quot;
    outputImagePath := &quot;./out.jpg&quot;

    imageBytes, err := ioutil.ReadFile(imagePath)
    if err != nil {
        fmt.Println(&quot;Error reading image file:&quot;, err)
        return
    }
    imageData := base64.StdEncoding.EncodeToString(imageBytes)

    payload := map[string]string{&quot;image&quot;: imageData}
    payloadBytes, err := json.Marshal(payload)
    if err != nil {
        fmt.Println(&quot;Error marshaling payload:&quot;, err)
        return
    }

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

    body, err := ioutil.ReadAll(res.Body)
    if err != nil {
        fmt.Println(&quot;Error reading response body:&quot;, err)
        return
    }
    type Response struct {
        Result struct {
            Image      string   `json:&quot;image&quot;`
            Categories []map[string]interface{} `json:&quot;categories&quot;`
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
    fmt.Println(&quot;\nCategories:&quot;)
    for _, category := range respData.Result.Categories {
        fmt.Println(category)
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
    static readonly string API_URL = &quot;http://localhost:8080/image-classification&quot;;
    static readonly string imagePath = &quot;./demo.jpg&quot;;
    static readonly string outputImagePath = &quot;./out.jpg&quot;;

    static async Task Main(string[] args)
    {
        var httpClient = new HttpClient();

        byte[] imageBytes = File.ReadAllBytes(imagePath);
        string image_data = Convert.ToBase64String(imageBytes);

        var payload = new JObject{ { &quot;image&quot;, image_data } };
        var content = new StringContent(payload.ToString(), Encoding.UTF8, &quot;application/json&quot;);

        HttpResponseMessage response = await httpClient.PostAsync(API_URL, content);
        response.EnsureSuccessStatusCode();

        string responseBody = await response.Content.ReadAsStringAsync();
        JObject jsonResponse = JObject.Parse(responseBody);

        string base64Image = jsonResponse[&quot;result&quot;][&quot;image&quot;].ToString();
        byte[] outputImageBytes = Convert.FromBase64String(base64Image);

        File.WriteAllBytes(outputImagePath, outputImageBytes);
        Console.WriteLine($&quot;Output image saved at {outputImagePath}&quot;);
        Console.WriteLine(&quot;\nCategories:&quot;);
        Console.WriteLine(jsonResponse[&quot;result&quot;][&quot;categories&quot;].ToString());
    }
}
</code></pre></details>

<details><summary>Node.js</summary>

<pre><code class="language-js">const axios = require('axios');
const fs = require('fs');

const API_URL = 'http://localhost:8080/image-classification'
const imagePath = './demo.jpg'
const outputImagePath = &quot;./out.jpg&quot;;

let config = {
   method: 'POST',
   maxBodyLength: Infinity,
   url: API_URL,
   data: JSON.stringify({
    'image': encodeImageToBase64(imagePath)
  })
};

function encodeImageToBase64(filePath) {
  const bitmap = fs.readFileSync(filePath);
  return Buffer.from(bitmap).toString('base64');
}

axios.request(config)
.then((response) =&gt; {
    const result = response.data[&quot;result&quot;];
    const imageBuffer = Buffer.from(result[&quot;image&quot;], 'base64');
    fs.writeFile(outputImagePath, imageBuffer, (err) =&gt; {
      if (err) throw err;
      console.log(`Output image saved at ${outputImagePath}`);
    });
    console.log(&quot;\nCategories:&quot;);
    console.log(result[&quot;categories&quot;]);
})
.catch((error) =&gt; {
  console.log(error);
});
</code></pre></details>
<details><summary>PHP</summary>

<pre><code class="language-php">&lt;?php

$API_URL = &quot;http://localhost:8080/image-classification&quot;;
$image_path = &quot;./demo.jpg&quot;;
$output_image_path = &quot;./out.jpg&quot;;

$image_data = base64_encode(file_get_contents($image_path));
$payload = array(&quot;image&quot; =&gt; $image_data);

$ch = curl_init($API_URL);
curl_setopt($ch, CURLOPT_POST, true);
curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($payload));
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
$response = curl_exec($ch);
curl_close($ch);

$result = json_decode($response, true)[&quot;result&quot;];
file_put_contents($output_image_path, base64_decode($result[&quot;image&quot;]));
echo &quot;Output image saved at &quot; . $output_image_path . &quot;\n&quot;;
echo &quot;\nCategories:\n&quot;;
print_r($result[&quot;categories&quot;]);
?&gt;
</code></pre></details>

</details>
<br/>

📱 <b>Edge Deployment</b>: Edge deployment is a method that places computing and data processing functions on user devices themselves, allowing devices to process data directly without relying on remote servers. PaddleX supports deploying models on edge devices such as Android. For detailed edge deployment procedures, refer to the [PaddleX Edge Deployment Guide](../../../pipeline_deploy/edge_deploy.en.md).
You can choose the appropriate deployment method for your model pipeline based on your needs and proceed with subsequent AI application integration.


## 4. Custom Development

If the default model weights provided by the General Image Recognition Pipeline do not meet your expectations in terms of precision or speed. You can further **fine-tune** the existing models using **your own data from specific domains or application scenarios** to enhance the recognition performance of the pipeline in your context.

### 4.1 Model Fine-Tuning

Since the General Image Recognition Pipeline consists of two modules (the mainbody detection module and the image feature module), the suboptimal performance of the pipeline may stem from either module.

You can analyze images with poor recognition results. After analysising, if you find that many mainbody objects are not detected, it may indicate deficiencies in the mainbody detection model. You need to refer to the [Custom Development](../../../module_usage/tutorials/cv_modules/mainbody_detection.en.md#custom-development) section in the [Object Detection Module Development Tutorial](../../../module_usage/tutorials/cv_modules/mainbody_detection.en.md) and use your private dataset to fine-tune the mainbody detection model. If there are mismatches in the detected mainbody objects, it suggests that the image feature model requires further improvement. You should refer to the [Custom Development](../../../module_usage/tutorials/cv_modules/image_feature.md#custom-development) section in the [Image Feature Module Development Tutorial](../../../module_usage/tutorials/cv_modules/image_feature.en.md) and fine-tune the image feature model.

### 4.2 Model Application

After you complete the fine-tuning training with your private dataset, you will obtain local model files.

To use the fine-tuned model, you only need to modify the pipeline configuration file by replacing with the paths to your fine-tuned model:

```yaml
Pipeline:
  device: "gpu:0"
  det_model: "./PP-ShiTuV2_det_infer/"        # Can be modified to the local path of the fine-tuned mainbody detection model
  rec_model: "./PP-ShiTuV2_rec_infer/"        # Can be modified to the local path of the fine-tuned image feature model
  det_batch_size: 1
  rec_batch_size: 1
  device: gpu
```
Subsequently, refer to the command-line method or Python script method in [2.2 Local Experience](#22-local-experience) to load the modified pipeline configuration file.

## 5. Multi-Hardware Support

PaddleX supports various mainstream hardware devices such as NVIDIA GPUs, Kunlun XPU, Ascend NPU, and Cambricon MLU. **Simply by modifying the `--device` parameter**, seamless switching between different hardware can be achieved.

For example, when running the General Image Recognition Pipeline using Python and changing the running device from an NVIDIA GPU to an Ascend NPU, you only need to modify the `device` in the script to `npu`:

```python
from paddlex import create_pipeline

pipeline = create_pipeline(
    pipeline="PP-ShiTuV2",
    device="npu:0" # gpu:0 --> npu:0
)
```

If you want to use the General Image Recognition Pipeline on more types of hardware, please refer to the [PaddleX Multi-Device Usage Guide](../../../other_devices_support/multi_devices_use_guide.en.md).
