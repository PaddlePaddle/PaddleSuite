---
comments: true
---

# General Table Recognition Pipeline Tutorial

## 1. Introduction to the General Table Recognition Pipeline
Table recognition is a technology that automatically identifies and extracts table content and its structure from documents or images. It is widely used in data entry, information retrieval, and document analysis. By leveraging computer vision and machine learning algorithms, table recognition can convert complex table information into editable formats, facilitating further data processing and analysis for users.

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/table_recognition/01.png">


<b>The General Table Recognition Pipeline comprises modules for table structure recognition, layout analysis, text detection, and text recognition.</b>

<b>If you prioritize model accuracy, choose a model with higher accuracy. If you prioritize inference speed, select a model with faster inference. If you prioritize model size, choose a model with a smaller storage footprint.</b>

<details><summary> 👉Model List Details</summary>

<p><b>Table Recognition Module Models</b>:</p>
<table>
<tr>
<th>Model</th>
<th>Accuracy (%)</th>
<th>GPU Inference Time (ms)</th>
<th>CPU Inference Time (ms)</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
<tr>
<td>SLANet</td>
<td>59.52</td>
<td>522.536</td>
<td>1845.37</td>
<td>6.9 M</td>
<td rowspan="1">SLANet is a table structure recognition model developed by Baidu PaddleX Team. The model significantly improves the accuracy and inference speed of table structure recognition by adopting a CPU-friendly lightweight backbone network PP-LCNet, a high-low-level feature fusion module CSP-PAN, and a feature decoding module SLA Head that aligns structural and positional information.</td>
</tr>
</tr>
<tr>
<td>SLANet_plus</td>
<td>63.69</td>
<td>522.536</td>
<td>1845.37</td>
<td>6.9 M</td>
<td rowspan="1">
SLANet_plus is an enhanced version of SLANet, a table structure recognition model developed by Baidu PaddleX Team. Compared to SLANet, SLANet_plus significantly improves its recognition capabilities for wireless and complex tables, while reducing the model's sensitivity to the accuracy of table localization. Even when there are offsets in table localization, it can still perform relatively accurate recognition.
</td>
</tr>
</table>

<p><b>Note: The above accuracy metrics are measured on PaddleX's internal self-built English table recognition dataset. All GPU inference times are based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speeds are based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b></p>
<p><b>Layout Analysis Module Models</b>:</p>
<table>
<thead>
<tr>
<th>Model</th>
<th>mAP(0.5) (%)</th>
<th>GPU Inference Time (ms)</th>
<th>CPU Inference Time (ms)</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>PicoDet_layout_1x</td>
<td>86.8</td>
<td>13.0</td>
<td>91.3</td>
<td>7.4</td>
<td>An efficient layout area localization model trained on the PubLayNet dataset based on PicoDet-1x can locate five types of areas, including text, titles, tables, images, and lists.</td>
</tr>
<tr>
<td>PicoDet-S_layout_3cls</td>
<td>87.1</td>
<td>13.5</td>
<td>45.8</td>
<td>4.8</td>
<td>An high-efficient layout area localization model trained on a self-constructed dataset based on PicoDet-S for scenarios such as Chinese and English papers, magazines, and research reports includes three categories: tables, images, and seals.</td>
</tr>
<tr>
<td>PicoDet-S_layout_17cls</td>
<td>70.3</td>
<td>13.6</td>
<td>46.2</td>
<td>4.8</td>
<td>A high-efficient layout area localization model trained on a self-constructed dataset based on PicoDet-S_layout_17cls for scenarios such as Chinese and English papers, magazines, and research reports includes 17 common layout categories, namely: paragraph titles, images, text, numbers, abstracts, content, chart titles, formulas, tables, table titles, references, document titles, footnotes, headers, algorithms, footers, and seals.</td>
</tr>
<tr>
<td>PicoDet-L_layout_3cls</td>
<td>89.3</td>
<td>15.7</td>
<td>159.8</td>
<td>22.6</td>
<td>An efficient layout area localization model trained on a self-constructed dataset based on PicoDet-L for scenarios such as Chinese and English papers, magazines, and research reports includes three categories: tables, images, and seals.</td>
</tr>
<tr>
<td>PicoDet-L_layout_17cls</td>
<td>79.9</td>
<td>17.2</td>
<td>160.2</td>
<td>22.6</td>
<td>A efficient layout area localization model trained on a self-constructed dataset based on PicoDet-L_layout_17cls for scenarios such as Chinese and English papers, magazines, and research reports includes 17 common layout categories, namely: paragraph titles, images, text, numbers, abstracts, content, chart titles, formulas, tables, table titles, references, document titles, footnotes, headers, algorithms, footers, and seals.</td>
</tr>
<tr>
<td>RT-DETR-H_layout_3cls</td>
<td>95.9</td>
<td>114.6</td>
<td>3832.6</td>
<td>470.1</td>
<td>A high-precision layout area localization model trained on a self-constructed dataset based on RT-DETR-H for scenarios such as Chinese and English papers, magazines, and research reports includes three categories: tables, images, and seals.</td>
</tr>
<tr>
<td>RT-DETR-H_layout_17cls</td>
<td>92.6</td>
<td>115.1</td>
<td>3827.2</td>
<td>470.2</td>
<td>A high-precision layout area localization model trained on a self-constructed dataset based on RT-DETR-H for scenarios such as Chinese and English papers, magazines, and research reports includes 17 common layout categories, namely: paragraph titles, images, text, numbers, abstracts, content, chart titles, formulas, tables, table titles, references, document titles, footnotes, headers, algorithms, footers, and seals.</td>
</tr>
</tbody>
</table>
<p><b>Note: The above accuracy metrics are evaluated on PaddleX's self-built layout analysis dataset containing 10,000 images. All GPU inference times are based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speeds are based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b></p>
<p><b>Text Detection Module Models</b>:</p>
<table>
<thead>
<tr>
<th>Model Name</th>
<th>Detection Hmean (%)</th>
<th>GPU Inference Time (ms)</th>
<th>CPU Inference Time (ms)</th>
<th>Model Size (M)</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-OCRv4_mobile_det</td>
<td>77.79</td>
<td>10.6923</td>
<td>120.177</td>
<td>4.2 M</td>
</tr>
<tr>
<td>PP-OCRv4_server_det</td>
<td>82.69</td>
<td>83.3501</td>
<td>2434.01</td>
<td>100.1M</td>
</tr>
</tbody>
</table></details>

## 2. Quick Start
PaddleX's pre-trained model pipelines allow for quick experience of their effects. You can experience the effects of the General Image Classification pipeline online or locally using command line or Python.

### 2.1 Online Experience
You can [experience online](https://aistudio.baidu.com/community/app/91661/webUI) the effects of the General Table Recognition pipeline by using the demo images provided by the official. For example:

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/table_recognition/02.png">

If you are satisfied with the pipeline's performance, you can directly integrate and deploy it. If not, you can also use your private data to <b>fine-tune the models in the pipeline online</b>.

### 2.2 Local Experience
Before using the General Table Recognition pipeline locally, ensure you have installed the PaddleX wheel package following the [PaddleX Local Installation Guide](../../../installation/installation.en.md).

### 2.1 Command Line Experience
Experience the effects of the table recognition pipeline with a single command:

Experience the image anomaly detection pipeline with a single command，Use the [test file](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/table_recognition.jpg), and replace `--input` with the local path to perform prediction.

```bash
paddlex --pipeline table_recognition --input table_recognition.jpg --device gpu:0
```
Parameter Explanation:

```
--pipeline: The name of the pipeline, here it's the table recognition pipeline.
--input: The local path or URL of the input image to be processed.
--device: The GPU index to use (e.g., gpu:0 for the first GPU, gpu:1,2 for the 1st and 2nd GPUs). CPU can also be selected (--device cpu).
```

When executing the above command, the default table recognition pipeline configuration file is loaded. If you need to customize the configuration file, you can execute the following command to obtain it:

<details><summary> 👉Click to expand</summary>

<pre><code class="language-bash">paddlex --get_pipeline_config table_recognition
</code></pre>
<p>After execution, the table recognition pipeline configuration file will be saved in the current directory. If you wish to customize the save location, you can execute the following command (assuming the custom save location is <code>./my_path</code>):</p>
<pre><code class="language-bash">paddlex --get_pipeline_config table_recognition --save_path ./my_path
</code></pre>
<p>After obtaining the pipeline configuration file, replace <code>--pipeline</code> with the configuration file save path to make the configuration file take effect. For example, if the configuration file save path is <code>./table_recognition.yaml</code>, simply execute:</p>
<pre><code class="language-bash">paddlex --pipeline ./table_recognition.yaml --input table_recognition.jpg --device gpu:0
</code></pre>
<p>Here, parameters like <code>--model</code> and <code>--device</code> do not need to be specified, as they will use the parameters in the configuration file. If they are still specified, the specified parameters will take precedence.</p></details>

After running, the result is:

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/table_recognition/03.png">

The visualized image not saved by default. You can customize the save path through `--save_path`, and then all results will be saved in the specified path.

### 2.2 Python Script Integration
A few lines of code are all you need to quickly perform inference with the pipeline. Taking the General Table Recognition pipeline as an example:

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="table_recognition")

output = pipeline.predict("table_recognition.jpg")
for res in output:
    res.print()  # Print the structured output of the prediction
    res.save_to_img("./output/")  # Save the results in img format
    res.save_to_xlsx("./output/")  # Save the results in Excel format
    res.save_to_html("./output/") # Save results in HTML format
```
The results are the same as those obtained through the command line.

In the above Python script, the following steps are executed:

（1）Instantiate the  production line object using `create_pipeline`: Specific parameter descriptions are as follows:

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
<td>The name of the production line or the path to the production line configuration file. If it is the name of the production line, it must be supported by PaddleX.</td>
<td><code>str</code></td>
<td>None</td>
</tr>
<tr>
<td><code>device</code></td>
<td>The device for production line model inference. Supports: "gpu", "cpu".</td>
<td><code>str</code></td>
<td><code>gpu</code></td>
</tr>
<tr>
<td><code>use_hpip</code></td>
<td>Whether to enable high-performance inference, only available if the production line supports it.</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
</tbody>
</table>
（2）Invoke the `predict` method of the  production line object for inference prediction: The `predict` method parameter is `x`, which is used to input data to be predicted, supporting multiple input methods, as shown in the following examples:

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
<td>Supports directly passing in Python variables, such as numpy.ndarray representing image data.</td>
</tr>
<tr>
<td>str</td>
<td>Supports passing in the path of the file to be predicted, such as the local path of an image file: <code>/root/data/img.jpg</code>.</td>
</tr>
<tr>
<td>str</td>
<td>Supports passing in the URL of the file to be predicted, such as the network URL of an image file: <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/table_recognition.jpg">Example</a>.</td>
</tr>
<tr>
<td>str</td>
<td>Supports passing in a local directory, which should contain files to be predicted, such as the local path: <code>/root/data/</code>.</td>
</tr>
<tr>
<td>dict</td>
<td>Supports passing in a dictionary type, where the key needs to correspond to a specific task, such as "img" for image classification tasks. The value of the dictionary supports the above types of data, for example: <code>{"img": "/root/data1"}</code>.</td>
</tr>
<tr>
<td>list</td>
<td>Supports passing in a list, where the list elements need to be of the above types of data, such as <code>[numpy.ndarray, numpy.ndarray], ["/root/data/img1.jpg", "/root/data/img2.jpg"], ["/root/data1", "/root/data2"], [{"img": "/root/data1"}, {"img": "/root/data2/img.jpg"}]</code>.</td>
</tr>
</tbody>
</table>
（3）Obtain the prediction results by calling the `predict` method: The `predict` method is a `generator`, so prediction results need to be obtained through iteration. The `predict` method predicts data in batches, so the prediction results are in the form of a list.

（4）Process the prediction results: The prediction result for each sample is of `dict` type and supports printing or saving to files, with the supported file types depending on the specific pipeline. For example:

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
<td>save_to_img</td>
<td>Save the results as an img format file</td>
<td><code>- save_path</code>: str, the path to save the file. When it's a directory, the saved file name will be consistent with the input file type;</td>
</tr>
<tr>
<td>save_to_html</td>
<td>Save the results as an html format file</td>
<td><code>- save_path</code>: str, the path to save the file. When it's a directory, the saved file name will be consistent with the input file type;</td>
</tr>
<tr>
<td>save_to_xlsx</td>
<td>Save the results as a spreadsheet format file</td>
<td><code>- save_path</code>: str, the path to save the file. When it's a directory, the saved file name will be consistent with the input file type;</td>
</tr>
</tbody>
</table>
Where `save_to_img` can save visualization results (including OCR result images, layout analysis result images, table structure recognition result images), `save_to_html` can directly save the table as an html file (including text and table formatting), and `save_to_xlsx` can save the table as an Excel format file (including text and formatting).

If you have a configuration file, you can customize the configurations of the image anomaly detection pipeline by simply modifying the `pipeline` parameter in the `create_pipeline` method to the path of the pipeline configuration file.

For example, if your configuration file is saved at `./my_path/table_recognition.yaml`, you only need to execute:

```python
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="./my_path/table_recognition.yaml")
output = pipeline.predict("table_recognition.jpg")
for res in output:
    res.print()  # Print the structured output of prediction
    res.save_to_img("./output/")  # Save results in img format
    res.save_to_xlsx("./output/")  # Save results in Excel format
    res.save_to_html("./output/") # Save results in HTML format
```

## 3. Development Integration/Deployment
If the pipeline meets your requirements for inference speed and accuracy in production, you can proceed with development integration/deployment.

If you need to directly apply the pipeline in your Python project, refer to the example code in [2.2 Python Script Integration](#22-python-script-integration).

Additionally, PaddleX provides three other deployment methods, detailed as follows:

🚀 <b>High-Performance Inference</b>: In actual production environments, many applications have stringent standards for deployment strategy performance metrics (especially response speed) to ensure efficient system operation and smooth user experience. To this end, PaddleX provides high-performance inference plugins that aim to deeply optimize model inference and pre/post-processing for significant end-to-end process acceleration. For detailed high-performance inference procedures, refer to the [PaddleX High-Performance Inference Guide](../../../pipeline_deploy/high_performance_inference.md).

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
<p>Operations provided by the service:</p>
<ul>
<li><b><code>infer</code></b></li>
</ul>
<p>Locate and recognize tables in images.</p>
<p><code>POST /table-recognition</code></p>
<ul>
<li>Request body properties:</li>
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
<p>Properties of <code>inferenceParams</code>:</p>
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
<td><code>maxLongSide</code></td>
<td><code>integer</code></td>
<td>During inference, if the length of the longer side of the input image for the text detection model is greater than <code>maxLongSide</code>, the image will be scaled so that the length of the longer side equals <code>maxLongSide</code>.</td>
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
<td><code>tables</code></td>
<td><code>array</code></td>
<td>Positions and contents of tables.</td>
</tr>
<tr>
<td><code>layoutImage</code></td>
<td><code>string</code></td>
<td>Layout area detection result image. The image is in JPEG format and encoded using Base64.</td>
</tr>
<tr>
<td><code>ocrImage</code></td>
<td><code>string</code></td>
<td>OCR result image. The image is in JPEG format and encoded using Base64.</td>
</tr>
</tbody>
</table>
<p>Each element in <code>tables</code> is an <code>object</code> with the following properties:</p>
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
<td><code>bbox</code></td>
<td><code>array</code></td>
<td>Table position. The elements in the array are the x-coordinate of the top-left corner, the y-coordinate of the top-left corner, the x-coordinate of the bottom-right corner, and the y-coordinate of the bottom-right corner of the bounding box, respectively.</td>
</tr>
<tr>
<td><code>html</code></td>
<td><code>string</code></td>
<td>Table recognition result in HTML format.</td>
</tr>
</tbody>
</table></details>

<details><summary>Multi-Language Service Invocation Examples</summary>

<details>
<summary>Python</summary>


<pre><code class="language-python">import base64
import requests

API_URL = &quot;http://localhost:8080/table-recognition&quot;
image_path = &quot;./demo.jpg&quot;
ocr_image_path = &quot;./ocr.jpg&quot;
layout_image_path = &quot;./layout.jpg&quot;

with open(image_path, &quot;rb&quot;) as file:
    image_bytes = file.read()
    image_data = base64.b64encode(image_bytes).decode(&quot;ascii&quot;)

payload = {&quot;image&quot;: image_data}

response = requests.post(API_URL, json=payload)

assert response.status_code == 200
result = response.json()[&quot;result&quot;]
with open(ocr_image_path, &quot;wb&quot;) as file:
    file.write(base64.b64decode(result[&quot;ocrImage&quot;]))
print(f&quot;Output image saved at {ocr_image_path}&quot;)
with open(layout_image_path, &quot;wb&quot;) as file:
    file.write(base64.b64decode(result[&quot;layoutImage&quot;]))
print(f&quot;Output image saved at {layout_image_path}&quot;)
print(&quot;\nDetected tables:&quot;)
print(result[&quot;tables&quot;])
</code></pre></details>

<details><summary>C++</summary>

<pre><code class="language-cpp">#include &lt;iostream&gt;
#include &quot;cpp-httplib/httplib.h&quot; // https://github.com/Huiyicc/cpp-httplib
#include &quot;nlohmann/json.hpp&quot; // https://github.com/nlohmann/json
#include &quot;base64.hpp&quot; // https://github.com/tobiaslocker/base64

int main() {
    httplib::Client client(&quot;localhost:8080&quot;);
    const std::string imagePath = &quot;./demo.jpg&quot;;
    const std::string ocrImagePath = &quot;./ocr.jpg&quot;;
    const std::string layoutImagePath = &quot;./layout.jpg&quot;;

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

    auto response = client.Post(&quot;/table-recognition&quot;, headers, body, &quot;application/json&quot;);

    if (response &amp;&amp; response-&gt;status == 200) {
        nlohmann::json jsonResponse = nlohmann::json::parse(response-&gt;body);
        auto result = jsonResponse[&quot;result&quot;];

        encodedImage = result[&quot;ocrImage&quot;];
        std::string decoded_string = base64::from_base64(encodedImage);
        std::vector&lt;unsigned char&gt; decodedOcrImage(decoded_string.begin(), decoded_string.end());
        std::ofstream outputOcrFile(ocrImagePath, std::ios::binary | std::ios::out);
        if (outputOcrFile.is_open()) {
            outputOcrFile.write(reinterpret_cast&lt;char*&gt;(decodedOcrImage.data()), decodedOcrImage.size());
            outputOcrFile.close();
            std::cout &lt;&lt; &quot;Output image saved at &quot; &lt;&lt; ocrImagePath &lt;&lt; std::endl;
        } else {
            std::cerr &lt;&lt; &quot;Unable to open file for writing: &quot; &lt;&lt; ocrImagePath &lt;&lt; std::endl;
        }

        encodedImage = result[&quot;layoutImage&quot;];
        decodedString = base64::from_base64(encodedImage);
        std::vector&lt;unsigned char&gt; decodedLayoutImage(decodedString.begin(), decodedString.end());
        std::ofstream outputLayoutFile(layoutImagePath, std::ios::binary | std::ios::out);
        if (outputLayoutFile.is_open()) {
            outputLayoutFile.write(reinterpret_cast&lt;char*&gt;(decodedLayoutImage.data()), decodedlayoutImage.size());
            outputLayoutFile.close();
            std::cout &lt;&lt; &quot;Output image saved at &quot; &lt;&lt; layoutImagePath &lt;&lt; std::endl;
        } else {
            std::cerr &lt;&lt; &quot;Unable to open file for writing: &quot; &lt;&lt; layoutImagePath &lt;&lt; std::endl;
        }

        auto tables = result[&quot;tables&quot;];
        std::cout &lt;&lt; &quot;\nDetected tables:&quot; &lt;&lt; std::endl;
        for (const auto&amp; table : tables) {
            std::cout &lt;&lt; table &lt;&lt; std::endl;
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
        String API_URL = &quot;http://localhost:8080/table-recognition&quot;;
        String imagePath = &quot;./demo.jpg&quot;;
        String ocrImagePath = &quot;./ocr.jpg&quot;;
        String layoutImagePath = &quot;./layout.jpg&quot;;

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
                String ocrBase64Image = result.get(&quot;ocrImage&quot;).asText();
                String layoutBase64Image = result.get(&quot;layoutImage&quot;).asText();
                JsonNode tables = result.get(&quot;tables&quot;);

                byte[] imageBytes = Base64.getDecoder().decode(ocrBase64Image);
                try (FileOutputStream fos = new FileOutputStream(ocrImagePath)) {
                    fos.write(imageBytes);
                }
                System.out.println(&quot;Output image saved at &quot; + ocrBase64Image);

                imageBytes = Base64.getDecoder().decode(layoutBase64Image);
                try (FileOutputStream fos = new FileOutputStream(layoutImagePath)) {
                    fos.write(imageBytes);
                }
                System.out.println(&quot;Output image saved at &quot; + layoutImagePath);

                System.out.println(&quot;\nDetected tables: &quot; + tables.toString());
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
    API_URL := &quot;http://localhost:8080/table-recognition&quot;
    imagePath := &quot;./demo.jpg&quot;
    ocrImagePath := &quot;./ocr.jpg&quot;
    layoutImagePath := &quot;./layout.jpg&quot;

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
            OcrImage      string   `json:&quot;ocrImage&quot;`
            LayoutImage      string   `json:&quot;layoutImage&quot;`
            Tables []map[string]interface{} `json:&quot;tables&quot;`
        } `json:&quot;result&quot;`
    }
    var respData Response
    err = json.Unmarshal([]byte(string(body)), &amp;respData)
    if err != nil {
        fmt.Println(&quot;Error unmarshaling response body:&quot;, err)
        return
    }

    ocrImageData, err := base64.StdEncoding.DecodeString(respData.Result.OcrImage)
    if err != nil {
        fmt.Println(&quot;Error decoding base64 image data:&quot;, err)
        return
    }
    err = ioutil.WriteFile(ocrImagePath, ocrImageData, 0644)
    if err != nil {
        fmt.Println(&quot;Error writing image to file:&quot;, err)
        return
    }
    fmt.Printf(&quot;Image saved at %s.jpg\n&quot;, ocrImagePath)

    layoutImageData, err := base64.StdEncoding.DecodeString(respData.Result.LayoutImage)
    if err != nil {
        fmt.Println(&quot;Error decoding base64 image data:&quot;, err)
        return
    }
    err = ioutil.WriteFile(layoutImagePath, layoutImageData, 0644)
    if err != nil {
        fmt.Println(&quot;Error writing image to file:&quot;, err)
        return
    }
    fmt.Printf(&quot;Image saved at %s.jpg\n&quot;, layoutImagePath)

    fmt.Println(&quot;\nDetected tables:&quot;)
    for _, table := range respData.Result.Tables {
        fmt.Println(table)
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
    static readonly string API_URL = &quot;http://localhost:8080/table-recognition&quot;;
    static readonly string imagePath = &quot;./demo.jpg&quot;;
    static readonly string ocrImagePath = &quot;./ocr.jpg&quot;;
    static readonly string layoutImagePath = &quot;./layout.jpg&quot;;

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

        string ocrBase64Image = jsonResponse[&quot;result&quot;][&quot;ocrImage&quot;].ToString();
        byte[] ocrImageBytes = Convert.FromBase64String(ocrBase64Image);
        File.WriteAllBytes(ocrImagePath, ocrImageBytes);
        Console.WriteLine($&quot;Output image saved at {ocrImagePath}&quot;);

        string layoutBase64Image = jsonResponse[&quot;result&quot;][&quot;layoutImage&quot;].ToString();
        byte[] layoutImageBytes = Convert.FromBase64String(layoutBase64Image);
        File.WriteAllBytes(layoutImagePath, layoutImageBytes);
        Console.WriteLine($&quot;Output image saved at {layoutImagePath}&quot;);

        Console.WriteLine(&quot;\nDetected tables:&quot;);
        Console.WriteLine(jsonResponse[&quot;result&quot;][&quot;tables&quot;].ToString());
    }
}
</code></pre></details>

<details><summary>Node.js</summary>

<pre><code class="language-js">const axios = require('axios');
const fs = require('fs');

const API_URL = 'http://localhost:8080/table-recognition'
const imagePath = './demo.jpg'
const ocrImagePath = &quot;./ocr.jpg&quot;;
const layoutImagePath = &quot;./layout.jpg&quot;;

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

    const imageBuffer = Buffer.from(result[&quot;ocrImage&quot;], 'base64');
    fs.writeFile(ocrImagePath, imageBuffer, (err) =&gt; {
      if (err) throw err;
      console.log(`Output image saved at ${ocrImagePath}`);
    });

    imageBuffer = Buffer.from(result[&quot;layoutImage&quot;], 'base64');
    fs.writeFile(layoutImagePath, imageBuffer, (err) =&gt; {
      if (err) throw err;
      console.log(`Output image saved at ${layoutImagePath}`);
    });

    console.log(&quot;\nDetected tables:&quot;);
    console.log(result[&quot;tables&quot;]);
})
.catch((error) =&gt; {
  console.log(error);
});
</code></pre></details>

<details><summary>PHP</summary>

<pre><code class="language-php">&lt;?php

$API_URL = &quot;http://localhost:8080/table-recognition&quot;;
$image_path = &quot;./demo.jpg&quot;;
$ocr_image_path = &quot;./ocr.jpg&quot;;
$layout_image_path = &quot;./layout.jpg&quot;;

$image_data = base64_encode(file_get_contents($image_path));
$payload = array(&quot;image&quot; =&gt; $image_data);

$ch = curl_init($API_URL);
curl_setopt($ch, CURLOPT_POST, true);
curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($payload));
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
$response = curl_exec($ch);
curl_close($ch);

$result = json_decode($response, true)[&quot;result&quot;];
file_put_contents($ocr_image_path, base64_decode($result[&quot;ocrImage&quot;]));
echo &quot;Output image saved at &quot; . $ocr_image_path . &quot;\n&quot;;

file_put_contents($layout_image_path, base64_decode($result[&quot;layoutImage&quot;]));
echo &quot;Output image saved at &quot; . $layout_image_path . &quot;\n&quot;;

echo &quot;\nDetected tables:\n&quot;;
print_r($result[&quot;tables&quot;]);

?&gt;
</code></pre></details>
</details>
<br/>

📱 <b>Edge Deployment</b>: Edge deployment is a method that places computing and data processing capabilities directly on user devices, allowing devices to process data without relying on remote servers. PaddleX supports deploying models on edge devices such as Android. For detailed edge deployment procedures, refer to the [PaddleX Edge Deployment Guide](../../../pipeline_deploy/edge_deploy.en.md).
Choose the appropriate deployment method for your model pipeline based on your needs, and proceed with subsequent AI application integration.

## 4. Custom Development
If the default model weights provided by the general table recognition pipeline do not meet your requirements for accuracy or speed in your specific scenario, you can try to further fine-tune the existing model using <b>your own domain-specific or application-specific data</b> to improve the recognition performance of the general table recognition pipeline in your scenario.

### 4.1 Model Fine-tuning
Since the general table recognition pipeline consists of four modules, unsatisfactory performance may stem from any of these modules.

Analyze images with poor recognition results and follow the rules below for analysis and model fine-tuning:

* If the detected table structure is incorrect (e.g., row and column recognition errors, incorrect cell positions), the table structure recognition module may be insufficient. You need to refer to the [Customization](../../../module_usage/tutorials/ocr_modules/table_structure_recognition.en.md#customization) section in the [Table Structure Recognition Module Development Tutorial](../../../module_usage/tutorials/ocr_modules/table_structure_recognition.en.md) and use your private dataset to fine-tune the table structure recognition model.
* If the table area is incorrectly located within the overall layout, the layout detection module may be insufficient. You need to refer to the [Customization](../../../module_usage/tutorials/ocr_modules/layout_detection.en.md#customization) section in the [Layout Detection Module Development Tutorial](../../../module_usage/tutorials/ocr_modules/layout_detection.en.md) and use your private dataset to fine-tune the layout detection model.
* If many texts are undetected (i.e., text miss detection), the text detection model may be insufficient. You need to refer to the [Customization](../../../module_usage/tutorials/ocr_modules/text_recognition.en.md#customization) section in the [Text Detection Module Development Tutorial](../../../module_usage/tutorials/ocr_modules/text_recognition.en.md) and use your private dataset to fine-tune the text detection model.
* If many detected texts contain recognition errors (i.e., the recognized text content does not match the actual text content), the text recognition model requires further improvement. You need to refer to the [Customization](../../../module_usage/tutorials/ocr_modules/table_structure_recognition.en.md#customization) section.
### 4.2 Model Application
After fine-tuning your model with a private dataset, you will obtain local model weights files.

To use the fine-tuned model weights, simply modify the production line configuration file by replacing the local path of the fine-tuned model weights to the corresponding location in the configuration file:

```python
......
 Pipeline:
  layout_model: PicoDet_layout_1x  # Can be modified to the local path of the fine-tuned model
  table_model: SLANet  # Can be modified to the local path of the fine-tuned model
  text_det_model: PP-OCRv4_mobile_det  # Can be modified to the local path of the fine-tuned model
  text_rec_model: PP-OCRv4_mobile_rec  # Can be modified to the local path of the fine-tuned model
  layout_batch_size: 1
  text_rec_batch_size: 1
  table_batch_size: 1
  device: "gpu:0"
......
```
Then, refer to the command line or Python script method in the local experience to load the modified production line configuration file.

## 5. Multi-Hardware Support
PaddleX supports various mainstream hardware devices such as NVIDIA GPU, Kunlun XPU, Ascend NPU, and Cambricon MLU. <b>Simply modify the `--device` parameter</b> to seamlessly switch between different hardware.

For example, if you use an NVIDIA GPU for table recognition pipeline inference, the Python command is:

```bash
paddlex --pipeline table_recognition --input table_recognition.jpg --device gpu:0
```
At this time, if you want to switch the hardware to Ascend NPU, simply modify `--device` in the Python command to npu:

```bash
paddlex --pipeline table_recognition --input table_recognition.jpg --device npu:0
```
If you want to use the general table recognition pipeline on more types of hardware, please refer to the [PaddleX Multi-Hardware Usage Guide](../../../other_devices_support/multi_devices_use_guide.en.md).
