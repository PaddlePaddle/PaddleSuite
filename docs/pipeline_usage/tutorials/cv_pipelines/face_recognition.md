简体中文 | [English](face_recognition_en.md)

# 人脸识别产线使用教程

## 1. 人脸识别产线介绍
人脸识别任务是计算机视觉领域的重要组成部分，旨在通过分析和比较人脸特征，实现对个人身份的自动识别。该任务不仅需要检测图像中的人脸，还需要对人脸图像进行特征提取和匹配，从而在数据库中找到对应的身份信息。人脸识别广泛应用于安全认证、监控系统、社交媒体和智能设备等场景。

人脸识别产线是专注于解决人脸定位和识别任务的端到端串联系统，可以从图像中快速准确地定位人脸区域、提取人脸特征，并与特征库中预先建立的特征做检索比对，从而确认身份信息。

![](https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/pipelines/face_recognition/01.jpg)

**人脸识别产线中包含了人脸检测模块和人脸识别模块**，每个模块中包含了若干模型，具体使用哪些模型，您可以根据下边的 benchmark 数据来选择。**如您更考虑模型精度，请选择精度较高的模型，如您更考虑模型推理速度，请选择推理速度较快的模型，如您更考虑模型存储大小，请选择存储大小较小的模型**。

<details>
   <summary> 👉模型列表详情</summary>

**人脸检测模块：**

| 模型 | AP (%)<br>Easy/Medium/Hard | GPU推理耗时 (ms) | CPU推理耗时 | 模型存储大小 (M) | 介绍                          |
|--------------------------|-----------------|--------------|---------|------------|-----------------------------|
| BlazeFace                | 77.7/73.4/49.5  |              |         | 0.447      |  轻量高效的人脸检测模型               |
| BlazeFace-FPN-SSH        | 83.2/80.5/60.5  |              |         | 0.606      | BlazeFace的改进模型，增加FPN和SSH结构   |
| PicoDet_LCNet_x2_5_face	 | 93.7/90.7/68.1  |              |         | 28.9       | 基于PicoDet_LCNet_x2_5的人脸检测模型 |
| PP-YOLOE_plus-S_face     | 93.9/91.8/79.8  |              |         | 26.5       | 基于PP-YOLOE_plus-S的人脸检测模型    |

注：以上精度指标是在WIDER-FACE验证集上，以640
*640作为输入尺寸评估得到的。所有模型 GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为8，精度类型为 FP32。

**人脸识别模块：**

| 模型            | 输出特征维度 | Acc (%)<br>AgeDB-30/CFP-FP/LFW | GPU推理耗时 (ms) | CPU推理耗时 | 模型存储大小 (M) | 介绍                                  |
|---------------|--------|-------------------------------|--------------|---------|------------|-------------------------------------|
| MobileFaceNet | 128    | 96.28/96.71/99.58             |              |         | 4.1        | 基于MobileFaceNet在MS1Mv3数据集上训练的人脸识别模型 |
| ResNet50_face  | 512    | 98.12/98.56/99.77             |              |         | 87.2       | 基于ResNet50在MS1Mv3数据集上训练的人脸识别模型      |

注：以上精度指标是分别在 AgeDB-30、CFP-FP 和 LFW 数据集上测得的 Accuracy。所有模型 GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为8，精度类型为 FP32。

</details>

## 2. 快速开始
PaddleX 所提供的预训练的模型产线均可以快速体验效果，你可以在线体验人脸识别产线的效果，也可以在本地使用命令行或 Python 体验人脸识别产线的效果。

### 2.1 在线体验

暂不支持在线体验

### 2.2 本地体验
> ❗ 在本地使用人脸识别产线前，请确保您已经按照[PaddleX安装教程](../../../installation/installation.md)完成了PaddleX的wheel包安装。

#### 2.2.1 命令行方式体验

暂不支持命令行体验

默认使用内置的的通用图像识别产线配置文件，若您需要自定义配置文件，可执行如下命令获取：

<details>
   <summary> 👉点击展开</summary>

```bash
paddlex --get_pipeline_config face_recognition
```

执行后，通用图像识别产线配置文件将被保存在当前路径。若您希望自定义保存位置，可执行如下命令（假设自定义保存位置为` ./my_path`）：

```bash
paddlex --get_pipeline_config face_recognition --save_path ./my_path
```

</details>

#### 2.2.2 Python脚本方式集成
请下载[测试图像](https://paddle-model-ecology.bj.bcebos.com/paddlex/demo_data/friends1.jpg)进行测试。
在该产线的运行示例中需要预先构建人脸特征库，您可以参考如下指令下载官方提供的demo数据用来后续构建人脸特征库。
您可以参考下面的命令将 Demo 数据集下载到指定文件夹：

```bash
cd /path/to/paddlex
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/face_demo_gallery.tar
tar -xf ./face_demo_gallery.tar
```

若您希望用私有数据集建立人脸特征库，可以参考[2.3节 构建特征库的数据组织方式](#23-构建特征库的数据组织方式)。之后通过几行代码即可完成人脸特征库建立和人脸识别产线的快速推理。

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="face_recognition")

pipeline.build_index(data_root="face_demo_gallery", index_dir="face_gallery_index")

output = pipeline.predict("friends1.jpg", index_dir="face_gallery_index")
for res in output:
    res.print()
    res.save_to_img("./output/")
```

在上述 Python 脚本中，执行了如下几个步骤：

（1）实例化 `create_pipeline` 实例化 人脸识别 产线对象。具体参数说明如下：

|参数|参数说明|参数类型|默认值|
|-|-|-|-|
|`pipeline`|产线名称或是产线配置文件路径。如为产线名称，则必须为 PaddleX 所支持的产线。|`str`|无|
| `index_dir` | 产线推理预测所用的检索库文件所在的目录，如不传入该参数，则需要在`predict()`中指定`index_dir`。 |`str` | None |
|`device`|产线模型推理设备。支持：“gpu”，“cpu”。|`str`|`gpu`|
|`use_hpip`|是否启用高性能推理，仅当该产线支持高性能推理时可用。|`bool`|`False`|

（2）调用人脸识别产线对象的 `build_index` 方法，构建人脸特征库。具体参数为说明如下：

|参数|参数说明|参数类型|默认值|
|-|-|-|-|
|`data_root`|数据集的根目录，数据组织方式参考[2.3节 构建特征库的数据组织方式](#2.3-构建特征库的数据组织方式)|`str`|无|
|`index_dir`|特征库的保存路径。成功调用`build_index`方法后会在改路径下生成两个文件：<br> `"id_map.pkl"` 保存了图像ID与图像特征标签之间的映射关系；<br> `“vector.index”`存储了每张图像的特征向量|`str`|无|

（3）调用人脸识别产线对象的 `predict` 方法进行推理预测：`predict` 方法参数为`input`，用于输入待预测数据，支持多种输入方式，具体示例如下：

| 参数类型      | 参数说明                                                                                                  |
|---------------|-----------------------------------------------------------------------------------------------------------|
| Python Var    | 支持直接传入Python变量，如numpy.ndarray表示的图像数据。                                               |
| str         | 支持传入待预测数据文件路径，如图像文件的本地路径：`/root/data/img.jpg`。                                   |
| str           | 支持传入待预测数据文件URL，如图像文件的网络URL：[示例](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_001.png)。|
| str           | 支持传入本地目录，该目录下需包含待预测数据文件，如本地路径：`/root/data/`。                               |
| dict          | 支持传入字典类型，字典的key需与具体任务对应，如图像分类任务对应\"img\"，字典的val支持上述类型数据，例如：`{\"img\": \"/root/data1\"}`。|
| list          | 支持传入列表，列表元素需为上述类型数据，如`[numpy.ndarray, numpy.ndarray]，[\"/root/data/img1.jpg\", \"/root/data/img2.jpg\"]`，`[\"/root/data1\", \"/root/data2\"]`，`[{\"img\": \"/root/data1\"}, {\"img\": \"/root/data2/img.jpg\"}]`。|

另外，`predict`方法支持参数`index_dir`用于设置检索库：
| 参数类型 | 参数说明 |
| - | - |
| `index_dir` | 产线推理预测所用的检索库文件所在的目录，如不传入该参数，则默认使用在`create_pipeline()`中通过参数`index_dir`指定的检索库。 |`str` | None |

（4）调用`predict`方法获取预测结果：`predict` 方法为`generator`，因此需要通过调用获得预测结果，`predict`方法以batch为单位对数据进行预测。

（5）对预测结果进行处理：每个样本的预测结果均为`dict`类型，且支持打印，或保存为文件，支持保存的类型与具体产线相关，如：

| 方法         | 说明                        | 方法参数                                                                                               |
|--------------|-----------------------------|--------------------------------------------------------------------------------------------------------|
| print        | 打印结果到终端              | `- format_json`：bool类型，是否对输出内容进行使用json缩进格式化，默认为True；<br>`- indent`：int类型，json格式化设置，仅当format_json为True时有效，默认为4；<br>`- ensure_ascii`：bool类型，json格式化设置，仅当format_json为True时有效，默认为False； |
| save_to_json | 将结果保存为json格式的文件   | `- save_path`：str类型，保存的文件路径，当为目录时，保存文件命名与输入文件类型命名一致；<br>`- indent`：int类型，json格式化设置，默认为4；<br>`- ensure_ascii`：bool类型，json格式化设置，默认为False； |
| save_to_img  | 将结果保存为图像格式的文件  | `- save_path`：str类型，保存的文件路径，当为目录时，保存文件命名与输入文件类型命名一致； |

若您获取了配置文件，即可对人脸识别产线各项配置进行自定义，只需要修改 `create_pipeline` 方法中的 `pipeline` 参数值为产线配置文件路径即可。

例如，若您的配置文件保存在 `./my_path/face_recognition.yaml` ，则只需执行：

```python
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="./my_path/face_recognition.yaml", index_dir="face_gallery_index")

output = pipeline.predict("friends1.jpg")
for res in output:
    res.print()
    res.save_to_img("./output/")
```

#### 2.2.3 人脸特征库的添加和删除操作

若您希望将更多的人脸图像添加到特征库中，则可以调用 `append_index` 方法；删除人脸图像特征，则可以调用 `remove_index` 方法。

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="face_recognition")
pipeline.buile_index(data_root="face_demo_gallery", index_dir="face_gallery_index", index_type="IVF")
pipeline.append_index(data_root="face_demo_gallery", index_dir="face_gallery_index", index_type="IVF")
pipeline.remove_index(data_root="face_demo_gallery", index_dir="face_gallery_index", index_type="IVF")
```

上述方法参数说明如下：
|参数|参数说明|参数类型|默认值|
|-|-|-|-|
|`data_root`|要添加的数据集的根目录。数据组织方式与构建特征库时相同，参考[2.3节 构建特征库的数据组织方式](#2.3-构建特征库的数据组织方式)|`str`|无|
|`index_dir`|特征库的存储目录，在 `append_index` 和 `remove_index` 中，同时也是被修改（或删除）的特征库的路径，|`str`|无|
|`index_type`|支持 `HNSW32`、`IVF`、`Flat`。其中，`HNSW32` 检索速度较快且精度较高，但不支持 `remove_index()` 操作；`IVF` 检索速度较快但精度相对较低，支持 `append_index()` 和 `remove_index()` 操作；`Flat` 检索速度较低精度较高，支持 `append_index()` 和 `remove_index()` 操作。|`str`|`HNSW32`|
|`metric_type`|支持：`IP`，内积（Inner Product）；`L2`，欧几里得距离（Euclidean Distance）。|`str`|`IP`|

### 2.3 构建特征库的数据组织方式

PaddleX的人脸识别产线示例需要使用预先构建好的特征库进行人脸特征检索。如果您希望用私有数据构建人脸特征库，则需要按照如下方式组织数据：

```bash
data_root             # 数据集根目录，目录名称可以改变  
├── images            # 图像的保存目录，目录名称可以改变
│   ├── ID0           # 身份ID名字，最好是有意义的名字，比如人名
│   │   ├── xxx.jpg   # 图片，此处支持层级嵌套
│   │   ├── xxx.jpg   # 图片，此处支持层级嵌套
│   │       ...  
│   ├── ID1           # 身份ID名字，最好是有意义的名字，比如人名
│   │   ...
└── gallery.txt       # 特征库数据集标注文件，文件名称不可改变。每行给出待检索人脸图像路径和图像特征标签，使用空格分隔，内容举例：images/Chandler/Chandler00037.jpg Chandler
```
## 3. 开发集成/部署
如果人脸识别产线可以达到您对产线推理速度和精度的要求，您可以直接进行开发集成/部署。

若您需要将人脸识别产线直接应用在您的Python项目中，可以参考 [2.2.2 Python脚本方式](#222-python脚本方式集成)中的示例代码。

此外，PaddleX 也提供了其他三种部署方式，详细说明如下：

🚀 **高性能推理**：在实际生产环境中，许多应用对部署策略的性能指标（尤其是响应速度）有着较严苛的标准，以确保系统的高效运行与用户体验的流畅性。为此，PaddleX 提供高性能推理插件，旨在对模型推理及前后处理进行深度性能优化，实现端到端流程的显著提速，详细的高性能推理流程请参考[PaddleX高性能推理指南](../../../pipeline_deploy/high_performance_inference.md)。

☁️ **服务化部署**：服务化部署是实际生产环境中常见的一种部署形式。通过将推理功能封装为服务，客户端可以通过网络请求来访问这些服务，以获取推理结果。PaddleX 支持用户以低成本实现产线的服务化部署，详细的服务化部署流程请参考[PaddleX服务化部署指南](../../../pipeline_deploy/service_deploy.md)。

下面是API参考和多语言服务调用示例：

<details>
<summary>API参考</summary>

对于服务提供的所有操作：

- 响应体以及POST请求的请求体均为JSON数据（JSON对象）。
- 当请求处理成功时，响应状态码为`200`，响应体的属性如下：

    |名称|类型|含义|
    |-|-|-|
    |`errorCode`|`integer`|错误码。固定为`0`。|
    |`errorMsg`|`string`|错误说明。固定为`"Success"`。|

    响应体还可能有`result`属性，类型为`object`，其中存储操作结果信息。

- 当请求处理未成功时，响应体的属性如下：

    |名称|类型|含义|
    |-|-|-|
    |`errorCode`|`integer`|错误码。与响应状态码相同。|
    |`errorMsg`|`string`|错误说明。|

服务提供的操作如下：

- **`infer`**

    获取图像OCR结果。

    `POST /ocr`

    - 请求体的属性如下：

        |名称|类型|含义|是否必填|
        |-|-|-|-|
        |`image`|`string`|服务可访问的图像文件的URL或图像文件内容的Base64编码结果。|是|
        |`inferenceParams`|`object`|推理参数。|否|

        `inferenceParams`的属性如下：

        |名称|类型|含义|是否必填|
        |-|-|-|-|
        |`maxLongSide`|`integer`|推理时，若文本检测模型的输入图像较长边的长度大于`maxLongSide`，则将对图像进行缩放，使其较长边的长度等于`maxLongSide`。|否|

    - 请求处理成功时，响应体的`result`具有如下属性：

        |名称|类型|含义|
        |-|-|-|
        |`texts`|`array`|文本位置、内容和得分。|
        |`image`|`string`|OCR结果图，其中标注检测到的文本位置。图像为JPEG格式，使用Base64编码。|

        `texts`中的每个元素为一个`object`，具有如下属性：

        |名称|类型|含义|
        |-|-|-|
        |`poly`|`array`|文本位置。数组中元素依次为包围文本的多边形的顶点坐标。|
        |`text`|`string`|文本内容。|
        |`score`|`number`|文本识别得分。|

        `result`示例如下：

        ```json
        {
          "texts": [
            {
              "poly": [
                [
                  444,
                  244
                ],
                [
                  705,
                  244
                ],
                [
                  705,
                  311
                ],
                [
                  444,
                  311
                ]
              ],
              "text": "北京南站",
              "score": 0.9
            },
            {
              "poly": [
                [
                  992,
                  248
                ],
                [
                  1263,
                  251
                ],
                [
                  1263,
                  318
                ],
                [
                  992,
                  315
                ]
              ],
              "text": "天津站",
              "score": 0.5
            }
          ],
          "image": "xxxxxx"
        }
        ```

</details>

<details>
<summary>多语言调用服务示例</summary>

<details>
<summary>Python</summary>

```python
import base64
import requests

API_URL = "http://localhost:8080/ocr" # 服务URL
image_path = "./demo.jpg"
output_image_path = "./out.jpg"

# 对本地图像进行Base64编码
with open(image_path, "rb") as file:
    image_bytes = file.read()
    image_data = base64.b64encode(image_bytes).decode("ascii")

payload = {"image": image_data}  # Base64编码的文件内容或者图像URL

# 调用API
response = requests.post(API_URL, json=payload)

# 处理接口返回数据
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

    // 对本地图像进行Base64编码
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

    // 调用API
    auto response = client.Post("/ocr", headers, body, "application/json");
    // 处理接口返回数据
    if (response && response->status == 200) {
        nlohmann::json jsonResponse = nlohmann::json::parse(response->body);
        auto result = jsonResponse["result"];

        encodedImage = result["image"];
        std::string decodedString = base64::from_base64(encodedImage);
        std::vector<unsigned char> decodedImage(decodedString.begin(), decodedString.end());
        std::ofstream outputImage(outPutImagePath, std::ios::binary | std::ios::out);
        if (outputImage.is_open()) {
            outputImage.write(reinterpret_cast<char*>(decodedImage.data()), decodedImage.size());
            outputImage.close();
            std::cout << "Output image saved at " << outPutImagePath << std::endl;
        } else {
            std::cerr << "Unable to open file for writing: " << outPutImagePath << std::endl;
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
        String API_URL = "http://localhost:8080/ocr"; // 服务URL
        String imagePath = "./demo.jpg"; // 本地图像
        String outputImagePath = "./out.jpg"; // 输出图像

        // 对本地图像进行Base64编码
        File file = new File(imagePath);
        byte[] fileContent = java.nio.file.Files.readAllBytes(file.toPath());
        String imageData = Base64.getEncoder().encodeToString(fileContent);

        ObjectMapper objectMapper = new ObjectMapper();
        ObjectNode params = objectMapper.createObjectNode();
        params.put("image", imageData); // Base64编码的文件内容或者图像URL

        // 创建 OkHttpClient 实例
        OkHttpClient client = new OkHttpClient();
        MediaType JSON = MediaType.Companion.get("application/json; charset=utf-8");
        RequestBody body = RequestBody.Companion.create(params.toString(), JSON);
        Request request = new Request.Builder()
                .url(API_URL)
                .post(body)
                .build();

        // 调用API并处理接口返回数据
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

    // 对本地图像进行Base64编码
    imageBytes, err := ioutil.ReadFile(imagePath)
    if err != nil {
        fmt.Println("Error reading image file:", err)
        return
    }
    imageData := base64.StdEncoding.EncodeToString(imageBytes)

    payload := map[string]string{"image": imageData} // Base64编码的文件内容或者图像URL
    payloadBytes, err := json.Marshal(payload)
    if err != nil {
        fmt.Println("Error marshaling payload:", err)
        return
    }

    // 调用API
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

    // 处理接口返回数据
    body, err := ioutil.ReadAll(res.Body)
    if err != nil {
        fmt.Println("Error reading response body:", err)
        return
    }
    type Response struct {
        Result struct {
            Image      string   `json:"image"`
            Texts []map[string]interface{} `json:"texts"`
        } `json:"result"`
    }
    var respData Response
    err = json.Unmarshal([]byte(string(body)), &respData)
    if err != nil {
        fmt.Println("Error unmarshaling response body:", err)
        return
    }

    outputImageData, err := base64.StdEncoding.DecodeString(respData.Result.Image)
    if err != nil {
        fmt.Println("Error decoding base64 image data:", err)
        return
    }
    err = ioutil.WriteFile(outputImagePath, outputImageData, 0644)
    if err != nil {
        fmt.Println("Error writing image to file:", err)
        return
    }
    fmt.Printf("Image saved at %s.jpg\n", outputImagePath)
    fmt.Println("\nDetected texts:")
    for _, text := range respData.Result.Texts {
        fmt.Println(text)
    }
}
```

</details>

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

        // 对本地图像进行Base64编码
        byte[] imageBytes = File.ReadAllBytes(imagePath);
        string image_data = Convert.ToBase64String(imageBytes);

        var payload = new JObject{ { "image", image_data } }; // Base64编码的文件内容或者图像URL
        var content = new StringContent(payload.ToString(), Encoding.UTF8, "application/json");

        // 调用API
        HttpResponseMessage response = await httpClient.PostAsync(API_URL, content);
        response.EnsureSuccessStatusCode();

        // 处理接口返回数据
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

const API_URL = 'http://localhost:8080/ocr'
const imagePath = './demo.jpg'
const outputImagePath = "./out.jpg";

let config = {
   method: 'POST',
   maxBodyLength: Infinity,
   url: API_URL,
   data: JSON.stringify({
    'image': encodeImageToBase64(imagePath)  // Base64编码的文件内容或者图像URL
  })
};

// 对本地图像进行Base64编码
function encodeImageToBase64(filePath) {
  const bitmap = fs.readFileSync(filePath);
  return Buffer.from(bitmap).toString('base64');
}

// 调用API
axios.request(config)
.then((response) => {
    // 处理接口返回数据
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

$API_URL = "http://localhost:8080/ocr"; // 服务URL
$image_path = "./demo.jpg";
$output_image_path = "./out.jpg";

// 对本地图像进行Base64编码
$image_data = base64_encode(file_get_contents($image_path));
$payload = array("image" => $image_data); // Base64编码的文件内容或者图像URL

// 调用API
$ch = curl_init($API_URL);
curl_setopt($ch, CURLOPT_POST, true);
curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($payload));
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
$response = curl_exec($ch);
curl_close($ch);

// 处理接口返回数据
$result = json_decode($response, true)["result"];
file_put_contents($output_image_path, base64_decode($result["image"]));
echo "Output image saved at " . $output_image_path . "\n";
echo "\nDetected texts:\n";
print_r($result["texts"]);

?>
```

</details>
</details>
<br/>

📱 **端侧部署**：端侧部署是一种将计算和数据处理功能放在用户设备本身上的方式，设备可以直接处理数据，而不需要依赖远程的服务器。PaddleX 支持将模型部署在 Android 等端侧设备上，详细的端侧部署流程请参考[PaddleX端侧部署指南](../../../pipeline_deploy/edge_deploy.md)。
您可以根据需要选择合适的方式部署模型产线，进而进行后续的 AI 应用集成。


## 4. 二次开发
如果 人脸识别 产线提供的默认模型权重在您的场景中，精度或速度不满意，您可以尝试利用**您自己拥有的特定领域或应用场景的数据**对现有模型进行进一步的**微调**，以提升通用该产线的在您的场景中的识别效果。

### 4.1 模型微调
由于人脸识别产线包含两个模块（人脸检测和人脸识别），模型产线的效果不及预期可能来自于其中任何一个模块。

您可以对识别效果差的图片进行分析，如果在分析过程中发现有较多的人脸未被检测出来，那么可能是人脸检测模型存在不足，您需要参考[人脸检测模块开发教程](../../../module_usage/tutorials/cv_modules/face_detection.md)中的[二次开发](../../../module_usage/tutorials/cv_modules/face_detection.md#四二次开发)章节，使用您的私有数据集对人脸检测模型进行微调；如果在已检测到的人脸出现匹配错误，这表明人脸识别模型需要进一步改进，您需要参考[人脸识别模块开发教程](../../../module_usage/tutorials/cv_modules/face_recognition.md)中的[二次开发](../../../module_usage/tutorials/cv_modules/face_recognition.md#四二次开发)章节,对人脸识别模型进行微调。

### 4.2 模型应用
当您使用私有数据集完成微调训练后，可获得本地模型权重文件。

若您需要使用微调后的模型权重，只需对产线配置文件做修改，将微调后模型权重的本地路径替换至产线配置文件中的对应位置即可：

```bash

......
Pipeline:
  device: "gpu:0"
  det_model: "BlazeFace"        #可修改为微调后人脸检测模型的本地路径
  rec_model: "MobileFaceNet"    #可修改为微调后人脸识别模型的本地路径
  det_batch_size: 1
  rec_batch_size: 1
  device: gpu
......
```
随后， 参考[2.2 本地体验](#22-本地体验)中的命令行方式或Python脚本方式，加载修改后的产线配置文件即可。
注：目前暂不支持为人脸检测和人脸识别模型设置单独的batch_size。

##  5. 多硬件支持
PaddleX 支持英伟达 GPU、昆仑芯 XPU、昇腾 NPU和寒武纪 MLU 等多种主流硬件设备，**仅需修改 `--device`参数**即可完成不同硬件之间的无缝切换。

例如，使用Python运行人脸识别线时，将运行设备从英伟达 GPU 更改为昇腾 NPU，仅需将脚本中的 `device` 修改为 npu 即可：

```python
from paddlex import create_pipeline
from paddlex import create_pipeline

pipeline = create_pipeline(
    pipeline="face_recognition",
    device="npu:0" # gpu:0 --> npu:0
    )
```
若您想在更多种类的硬件上使用人脸识别产线，请参考[PaddleX多硬件使用指南](../../../other_devices_support/multi_devices_use_guide.md)。