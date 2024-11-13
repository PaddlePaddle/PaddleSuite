---
comments: true
typora-copy-images-to: images
hide:
  - navigation
  - toc
---

### 🛠️ Installation

> ❗Before installing PaddleX, please ensure you have a basic <b>Python environment</b> (Note: Currently supports Python 3.8 to Python 3.10, with more Python versions being adapted).

* <b>Installing PaddlePaddle</b>

```bash
# cpu
python -m pip install paddlepaddle==3.0.0b2 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

# gpu，该命令仅适用于 CUDA 版本为 11.8 的机器环境
python -m pip install paddlepaddle-gpu==3.0.0b2 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# gpu，该命令仅适用于 CUDA 版本为 12.3 的机器环境
python -m pip install paddlepaddle-gpu==3.0.0b2 -i https://www.paddlepaddle.org.cn/packages/stable/cu123/
```
> ❗For more PaddlePaddle versions, please refer to the [PaddlePaddle official website](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation./docs/zh/install/pip/linux-pip.html).

* <b>Installing PaddleX</b>

```bash
pip install https://paddle-model-ecology.bj.bcebos.com/paddlex/whl/paddlex-3.0.0b2-py3-none-any.whl
```

> ❗For more installation methods, refer to the [PaddleX Installation Guide](https://paddlepaddle.github.io/PaddleX/latest/en/installation/installation.html).


### 💻 CLI Usage

One command can quickly experience the pipeline effect, the unified CLI format is:

```bash
paddlex --pipeline [Pipeline Name] --input [Input Image] --device [Running Device]
```

You only need to specify three parameters:
* `pipeline`: The name of the pipeline
* `input`: The local path or URL of the input image to be processed
* `device`: The GPU number used (for example, `gpu:0` means using the 0th GPU), you can also choose to use the CPU (`cpu`)

For example, using the  OCR pipeline:
```bash
paddlex --pipeline OCR --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png  --device gpu:0
```

<pre><code class="language-bash">{
'input_path': '/root/.paddlex/predict_input/general_ocr_002.png',
'dt_polys': [array([[161,  27],
       [353,  22],
       [354,  69],
       [162,  74]], dtype=int16), array([[426,  26],
       [657,  21],
       [657,  58],
       [426,  62]], dtype=int16), array([[702,  18],
       [822,  13],
       [824,  57],
       [704,  62]], dtype=int16), array([[341, 106],
       [405, 106],
       [405, 128],
       [341, 128]], dtype=int16)
       ...],
'dt_scores': [0.758478200014338, 0.7021546472698513, 0.8536622648391111, 0.8619181462164781, 0.8321051217096188, 0.8868756173427551, 0.7982964727675609, 0.8289939036796322, 0.8289428877522524, 0.8587063317632897, 0.7786755892491615, 0.8502032769081344, 0.8703346500042997, 0.834490931790065, 0.908291103353393, 0.7614978661708064, 0.8325774055997542, 0.7843421347676149, 0.8680889482955594, 0.8788859304537682, 0.8963341277518075, 0.9364654810069546, 0.8092413027028257, 0.8503743089091863, 0.7920740420391101, 0.7592224394793805, 0.7920547400069311, 0.6641757962457888, 0.8650289477605955, 0.8079483304467047, 0.8532207681055275, 0.8913377034754717],
'rec_text': ['登机牌', 'BOARDING', 'PASS', '舱位', 'CLASS', '序号 SERIALNO.', '座位号', '日期 DATE', 'SEAT NO', '航班 FLIGHW', '035', 'MU2379', '始发地', 'FROM', '登机口', 'GATE', '登机时间BDT', '目的地TO', '福州', 'TAIYUAN', 'G11', 'FUZHOU', '身份识别IDNO', '姓名NAME', 'ZHANGQIWEI', 票号TKTNO', '张祺伟', '票价FARE', 'ETKT7813699238489/1', '登机口于起飞前10分钟关闭GATESCLOSE10MINUTESBEFOREDEPARTURETIME'],
'rec_score': [0.9985831379890442, 0.999696917533874512, 0.9985735416412354, 0.9842517971992493, 0.9383274912834167, 0.9943678975105286, 0.9419361352920532, 0.9221674799919128, 0.9555020928382874, 0.9870321154594421, 0.9664073586463928, 0.9988052248954773, 0.9979352355003357, 0.9985110759735107, 0.9943482875823975, 0.9991195797920227, 0.9936401844024658, 0.9974591135978699, 0.9743705987930298, 0.9980487823486328, 0.9874696135520935, 0.9900962710380554, 0.9952947497367859, 0.9950481653213501, 0.989926815032959, 0.9915552139282227, 0.9938777685165405, 0.997239887714386, 0.9963340759277344, 0.9936134815216064, 0.97223961353302]}
</code></pre>
<p>The visualization result is as follows:</p>
<p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/boardingpass.png"></p>

To use the command line for other pipelines, simply adjust the `pipeline` parameter to the name of the corresponding pipeline. Below are the commands for each pipeline:


<table>
<thead>
<tr>
<th>Pipeline Name</th>
<th>Command</th>
</tr>
</thead>
<tbody>
<tr>
<td>Image Classification</td>
<td><code>paddlex --pipeline image_classification --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg --device gpu:0</code></td>
</tr>
<tr>
<td>Object Detection</td>
<td><code>paddlex --pipeline object_detection --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_object_detection_002.png --device gpu:0</code></td>
</tr>
<tr>
<td>Instance Segmentation</td>
<td><code>paddlex --pipeline instance_segmentation --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_instance_segmentation_004.png --device gpu:0</code></td>
</tr>
<tr>
<td>Semantic Segmentation</td>
<td><code>paddlex --pipeline semantic_segmentation --input https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/application/semantic_segmentation/makassaridn-road_demo.png --device gpu:0</code></td>
</tr>
<tr>
<td>Image Multi-label Classification</td>
<td><code>paddlex --pipeline multi_label_image_classification --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg --device gpu:0</code></td>
</tr>
<tr>
<td>Small Object Detection</td>
<td><code>paddlex --pipeline small_object_detection --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/small_object_detection.jpg --device gpu:0</code></td>
</tr>
<tr>
<td>Image Anomaly Detection</td>
<td><code>paddlex --pipeline anomaly_detection --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/uad_grid.png --device gpu:0 </code></td>
</tr>
<tr>
<td>OCR</td>
<td><code>paddlex --pipeline OCR --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png --device gpu:0</code></td>
</tr>
<tr>
<td>Table Recognition</td>
<td><code>paddlex --pipeline table_recognition --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/table_recognition.jpg --device gpu:0</code></td>
</tr>
<tr>
<td>Layout Parsing</td>
<td><code>paddlex --pipeline layout_parsing --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/demo_paper.png --device gpu:0</code></td>
</tr>
<tr>
<td>Formula Recognition</td>
<td><code>paddlex --pipeline formula_recognition --input https://paddle-model-ecology.bj.bcebos.com/paddlex/demo_image/general_formula_recognition.png --device gpu:0</code></td>
</tr>
<tr>
<td>Seal Recognition</td>
<td><code>paddlex --pipeline seal_recognition --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/seal_text_det.png --device gpu:0</code></td>
</tr>
<tr>
<td>Time Series Forecasting</td>
<td><code>paddlex --pipeline ts_fc --input https://paddle-model-ecology.bj.bcebos.com/paddlex/ts/demo_ts/ts_fc.csv --device gpu:0</code></td>
</tr>
<tr>
<td>Time Series Anomaly Detection</td>
<td><code>paddlex --pipeline ts_ad --input https://paddle-model-ecology.bj.bcebos.com/paddlex/ts/demo_ts/ts_ad.csv --device gpu:0</code></td>
</tr>
<tr>
<td>Time Series Classification</td>
<td><code>paddlex --pipeline ts_cls --input https://paddle-model-ecology.bj.bcebos.com/paddlex/ts/demo_ts/ts_cls.csv --device gpu:0</code></td>
</tr>
</tbody>
</table>

### 📝 Python Script Usage

With just a few lines of code, you can quickly perform inference on a production line. The unified Python script format is as follows:
```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline=[PipelineName])
output = pipeline.predict([InputImageName])
for res in output:
    res.print()
    res.save_to_img("./output/")
    res.save_to_json("./output/")
```
The following steps are executed:

* `create_pipeline()` instantiates the production line object.
* An image is passed in and the `predict` method of the production line object is called for inference and prediction.
* The prediction results are processed.

For other production lines using the Python script, you only need to adjust the `pipeline` parameter of the `create_pipeline()` method to the corresponding production line name. Below is a list of each production line's corresponding parameter name and detailed usage explanation:

<table>
<thead>
<tr>
<th>Production Line Name</th>
<th>Corresponding Parameter</th>
<th>Detailed Explanation</th>
</tr>
</thead>
<tbody>
<tr>
<td>Document Scene Information Extraction v3</td>
<td><code>PP-ChatOCRv3-doc</code></td>
<td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_deploy/tutorials/information_extraction_pipelines/document_scene_information_extraction.html">Document Scene Information Extraction v3 Python Script Instructions</a></td>
</tr>
<tr>
<td>General Image Classification</td>
<td><code>image_classification</code></td>
<td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_deploy/tutorials/cv_pipelines/image_classification.html">General Image Classification Python Script Instructions</a></td>
</tr>
<tr>
<td>General Object Detection</td>
<td><code>object_detection</code></td>
<td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_deploy/tutorials/cv_pipelines/object_detection.html">General Object Detection Python Script Instructions</a></td>
</tr>
<tr>
<td>General Instance Segmentation</td>
<td><code>instance_segmentation</code></td>
<td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_deploy/tutorials/cv_pipelines/instance_segmentation.html">General Instance Segmentation Python Script Instructions</a></td>
</tr>
<tr>
<td>General Semantic Segmentation</td>
<td><code>semantic_segmentation</code></td>
<td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_deploy/tutorials/cv_pipelines/semantic_segmentation.html">General Semantic Segmentation Python Script Instructions</a></td>
</tr>
<tr>
<td>Image Multi-label Classification</td>
<td><code>multi_label_image_classification</code></td>
<td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_deploy/tutorials/cv_pipelines/image_multi_label_classification.html">Image Multi-label Classification Python Script Instructions</a></td>
</tr>
<tr>
<td>Small Object Detection</td>
<td><code>small_object_detection</code></td>
<td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_deploy/tutorials/cv_pipelines/small_object_detection.html">Small Object Detection Python Script Instructions</a></td>
</tr>
<tr>
<td>Image Anomaly Detection</td>
<td><code>anomaly_detection</code></td>
<td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_deploy/tutorials/cv_pipelines/image_anomaly_detection.html">Image Anomaly Detection Python Script Instructions</a></td>
</tr>
<tr>
<td>General OCR</td>
<td><code>OCR</code></td>
<td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_deploy/tutorials/ocr_pipelines/OCR.html">General OCR Python Script Instructions</a></td>
</tr>
<tr>
<td>General Table Recognition</td>
<td><code>table_recognition</code></td>
<td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_deploy/tutorials/ocr_pipelines/table_recognition.html">General Table Recognition Python Script Instructions</a></td>
</tr>
<tr>
<td>General Layout Parsing</td>
<td><code>layout_parsing</code></td>
<td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_deploy/tutorials/ocr_pipelines/layout_parsing.html">General Layout Parsing Python Script Instructions</a></td>
</tr>
<tr>
<td>Formula Recognition</td>
<td><code>formula_recognition</code></td>
<td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_deploy/tutorials/ocr_pipelines/formula_recognition.html">Formula Recognition Python Script Instructions</a></td>
</tr>
<tr>
<td>Seal Text Recognition</td>
<td><code>seal_recognition</code></td>
<td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_deploy/tutorials/ocr_pipelines/seal_recognition.html">Seal Text Recognition Python Script Instructions</a></td>
</tr>
<tr>
<td>Time Series Forecasting</td>
<td><code>ts_fc</code></td>
<td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_deploy/tutorials/time_series_pipelines/time_series_forecasting.html">Time Series Forecasting Python Script Instructions</a></td>
</tr>
<tr>
<td>Time Series Anomaly Detection</td>
<td><code>ts_ad</code></td>
<td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_deploy/tutorials/time_series_pipelines/time_series_anomaly_detection.html">Time Series Anomaly Detection Python Script Instructions</a></td>
</tr>
<tr>
<td>Time Series Classification</td>
<td><code>ts_cls</code></td>
<td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_deploy/tutorials/time_series_pipelines/time_series_classification.html">Time Series Classification Python Script Instructions</a></td>
</tr>
</tbody>
</table>
