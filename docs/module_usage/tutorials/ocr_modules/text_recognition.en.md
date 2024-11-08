---
comments: true
---

# Text Recognition Module Development Tutorial

## I. Overview
The text recognition module is the core component of an OCR (Optical Character Recognition) system, responsible for extracting text information from text regions within images. The performance of this module directly impacts the accuracy and efficiency of the entire OCR system. The text recognition module typically receives bounding boxes of text regions output by the text detection module as input. Through complex image processing and deep learning algorithms, it converts the text in images into editable and searchable electronic text. The accuracy of text recognition results is crucial for subsequent applications such as information extraction and data mining.

## II. Supported Model List

<table>
  <tr>
    <th>Model</th>
    <th>Recognition Avg Accuracy(%)</th>
    <th>GPU Inference Time (ms)</th>
    <th>CPU Inference Time (ms)</th>
    <th>Model Size (M)</th>
    <th>Description</th>
  </tr>
   <tr>
        <td>PP-OCRv4_mobile_rec</td>
        <td>78.20</td>
        <td>7.95018</td>
        <td>46.7868</td>
        <td>10.6 M</td>
        <td rowspan="2">PP-OCRv4, developed by Baidu's PaddlePaddle Vision Team, is the next version of the PP-OCRv3 text recognition model. By introducing data augmentation schemes, GTC-NRTR guidance branches, and other strategies, it further improves text recognition accuracy without compromising model inference speed. The model offers both server and mobile versions to meet industrial needs in different scenarios.</td>
    </tr>
    <tr>
        <td>PP-OCRv4_server_rec </td>
        <td>79.20</td>
        <td>7.19439</td>
        <td>140.179</td>
        <td>71.2 M</td>
    </tr>
</table>

<b>Note: The evaluation set for the above accuracy metrics is PaddleOCR's self-built Chinese dataset, covering street scenes, web images, documents, handwriting, and more, with 1.1w images for text recognition. GPU inference time for all models is based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speed is based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b>

> ❗ The above list features the <b>2 core models</b> that the image classification module primarily supports. In total, this module supports <b>4 models</b>. The complete list of models is as follows:

<details><summary> 👉Model List Details</summary>

<table>
<tr>
<th>Model</th>
<th>Recognition Avg Accuracy(%)</th>
<th>GPU Inference Time (ms)</th>
<th>CPU Inference Time (ms)</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
<tr>
<td>PP-OCRv4_mobile_rec</td>
<td>78.20</td>
<td>7.95018</td>
<td>46.7868</td>
<td>10.6 M</td>
<td rowspan="2">PP-OCRv4, developed by Baidu's PaddlePaddle Vision Team, is the next version of the PP-OCRv3 text recognition model. By introducing data augmentation schemes, GTC-NRTR guidance branches, and other strategies, it further improves text recognition accuracy without compromising model inference speed. The model offers both server and mobile versions to meet industrial needs in different scenarios.</td>
</tr>
<tr>
<td>PP-OCRv4_server_rec </td>
<td>79.20</td>
<td>7.19439</td>
<td>140.179</td>
<td>71.2 M</td>
</tr>
</table>

<p><b>Note: The evaluation set for the above accuracy metrics is PaddleOCR's self-built Chinese dataset, covering street scenes, web images, documents, handwriting, and more, with 1.1w images for text recognition. GPU inference time for all models is based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speed is based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b></p>
<table>
<tr>
<th>Model</th>
<th>Recognition Avg Accuracy(%)</th>
<th>GPU Inference Time (ms)</th>
<th>CPU Inference Time</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
<tr>
<td>ch_SVTRv2_rec</td>
<td>68.81</td>
<td>8.36801</td>
<td>165.706</td>
<td>73.9 M</td>
<td rowspan="1">SVTRv2, a server-side text recognition model developed by the OpenOCR team at the Vision and Learning Lab (FVL) of Fudan University, also won first place in the OCR End-to-End Recognition Task of the PaddleOCR Algorithm Model Challenge. Its A-rank end-to-end recognition accuracy is 6% higher than PP-OCRv4.
</td>
</tr>
</table>

<p><b>Note: The evaluation set for the above accuracy metrics is the <a href="https://aistudio.baidu.com/competition/detail/1131/0/introduction">OCR End-to-End Recognition Task of the PaddleOCR Algorithm Model Challenge - Track 1</a> A-rank. GPU inference time for all models is based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speed is based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b></p>
<table>
<tr>
<th>Model</th>
<th>Recognition Avg Accuracy(%)</th>
<th>GPU Inference Time (ms)</th>
<th>CPU Inference Time</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
<tr>
<td>ch_RepSVTR_rec</td>
<td>65.07</td>
<td>10.5047</td>
<td>51.5647</td>
<td>22.1 M</td>
<td rowspan="1">  RepSVTR, a mobile text recognition model based on SVTRv2, won first place in the OCR End-to-End Recognition Task of the PaddleOCR Algorithm Model Challenge. Its B-rank end-to-end recognition accuracy is 2.5% higher than PP-OCRv4, with comparable inference speed.</td>
</tr>
</table>

<p><b>Note: The evaluation set for the above accuracy metrics is the <a href="https://aistudio.baidu.com/competition/detail/1131/0/introduction">OCR End-to-End Recognition Task of the PaddleOCR Algorithm Model Challenge - Track 1</a> B-rank. GPU inference time for all models is based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speed is based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b></p></details>

## III. Quick Integration
Before quick integration, you need to install the PaddleX wheel package. For the installation method, please refer to the [PaddleX Local Installation Tutorial](../../../installation/installation.en.md). After installing the wheel package, a few lines of code can complete the inference of the text recognition module. You can switch models under this module freely, and you can also integrate the model inference of the text recognition module into your project.

Before running the following code, please download the [demo image](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_rec_001.png) to your local machine.

```python
from paddlex import create_model
model = create_model("PP-OCRv4_mobile_rec")
output = model.predict("general_ocr_rec_001.png", batch_size=1)
for res in output:
    res.print(json_format=False)
    res.save_to_img("./output/")
    res.save_to_json("./output/res.json")
```
For more information on using PaddleX's single-model inference APIs, please refer to the [PaddleX Single-Model Python Script Usage Instructions](../../instructions/model_python_API.en.md).

## IV. Custom Development
If you are seeking higher accuracy from existing models, you can use PaddleX's custom development capabilities to develop better  text recognition models. Before using PaddleX to develop text recognition models, please ensure that you have installed the relevant model training plugins for OCR in PaddleX. The installation process can be found in the custom development section of the [PaddleX Local Installation Guide](../../../installation/installation.en.md).

### 4.1 Data Preparation
Before model training, it is necessary to prepare the corresponding dataset for each task module. PaddleX provides a data validation function for each module, and <b>only data that passes the validation can be used for model training</b>. Additionally, PaddleX offers Demo datasets for each module, allowing you to complete subsequent development based on the officially provided Demo data. If you wish to use a private dataset for subsequent model training, you can refer to the [PaddleX Text Detection/Text Recognition Task Module Data Annotation Tutorial](../../../data_annotations/ocr_modules/text_detection_recognition.en.md).

#### 4.1.1 Download Demo Data
You can use the following commands to download the Demo dataset to a specified folder:

```bash
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/ocr_rec_dataset_examples.tar -P ./dataset
tar -xf ./dataset/ocr_rec_dataset_examples.tar -C ./dataset/
```

#### 4.1.2 Data Validation
A single command can complete data validation:

```bash
python main.py -c paddlex/configs/text_recognition/PP-OCRv4_mobile_rec.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/ocr_rec_dataset_examples
```
After executing the above command, PaddleX will validate the dataset and summarize its basic information. If the command runs successfully, it will print `Check dataset passed !` in the log. The validation results file is saved in `./output/check_dataset_result.json`, and related outputs are saved in the `./output/check_dataset` directory in the current directory, including visual examples of sample images and sample distribution histograms.

<details><summary>👉 <b>Validation Result Details (Click to Expand)</b></summary>

<p>The specific content of the validation result file is:</p>
<pre><code class="language-bash">{
  &quot;done_flag&quot;: true,
  &quot;check_pass&quot;: true,
  &quot;attributes&quot;: {
    &quot;train_samples&quot;: 4468,
    &quot;train_sample_paths&quot;: [
      &quot;../dataset/ocr_rec_dataset_examples/images/train_word_1.png&quot;,
      &quot;../dataset/ocr_rec_dataset_examples/images/train_word_10.png&quot;
    ],
    &quot;val_samples&quot;: 2077,
    &quot;val_sample_paths&quot;: [
      &quot;../dataset/ocr_rec_dataset_examples/images/val_word_1.png&quot;,
      &quot;../dataset/ocr_rec_dataset_examples/images/val_word_10.png&quot;
    ]
  },
  &quot;analysis&quot;: {
    &quot;histogram&quot;: &quot;check_dataset/histogram.png&quot;
  },
  &quot;dataset_path&quot;: &quot;./dataset/ocr_rec_dataset_examples&quot;,
  &quot;show_type&quot;: &quot;image&quot;,
  &quot;dataset_type&quot;: &quot;MSTextRecDataset&quot;
}
</code></pre>
<p>In the above validation result, <code>check_pass</code> being <code>true</code> indicates that the dataset format meets the requirements. Explanations for other indicators are as follows:</p>
<ul>
<li><code>attributes.train_samples</code>: The number of training set samples in this dataset is 4468;</li>
<li><code>attributes.val_samples</code>: The number of validation set samples in this dataset is 2077;</li>
<li><code>attributes.train_sample_paths</code>: A list of relative paths to the visualized training set samples in this dataset;</li>
<li><code>attributes.val_sample_paths</code>: A list of relative paths to the visualized validation set samples in this dataset;
Additionally, the dataset validation also analyzes the distribution of character length ratios in the dataset and generates a distribution histogram (histogram.png):</li>
</ul>
<p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/modules/text_recog/01.png"></p></details>

#### 4.1.3 Dataset Format Conversion/Dataset Splitting (Optional)


After completing data validation, you can convert the dataset format or re-split the training/validation ratio of the dataset by <b>modifying the configuration file</b> or <b>appending hyperparameters</b>.

<details><summary>👉 <b>Dataset Format Conversion/Dataset Splitting Details (Click to Expand)</b></summary>

<p><b>(1) Dataset Format Conversion</b></p>
<p>Text recognition does not currently support data conversion.</p>
<p><b>(2) Dataset Splitting</b></p>
<p>The parameters for dataset splitting can be set by modifying the <code>CheckDataset</code> section in the configuration file. Examples of some parameters in the configuration file are as follows:</p>
<ul>
<li><code>CheckDataset</code>:</li>
<li><code>split</code>:</li>
<li><code>enable</code>: Whether to re-split the dataset. Set to <code>True</code> to enable dataset splitting, default is <code>False</code>;</li>
<li><code>train_percent</code>: If re-splitting the dataset, set the percentage of the training set. The type is any integer between 0-100, and it must sum up to 100 with <code>val_percent</code>;
For example, if you want to re-split the dataset with a 90% training set and a 10% validation set, modify the configuration file as follows:</li>
</ul>
<pre><code class="language-bash">......
CheckDataset:
  ......
  split:
    enable: True
    train_percent: 90
    val_percent: 10
  ......
</code></pre>
<p>Then execute the command:</p>
<pre><code class="language-bash">python main.py -c paddlex/configs/text_recognition/PP-OCRv4_mobile_rec.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/ocr_rec_dataset_examples
</code></pre>
<p>After data splitting, the original annotation files will be renamed to <code>xxx.bak</code> in the original path.</p>
<p>The above parameters also support setting through appending command line arguments:</p>
<pre><code class="language-bash">python main.py -c paddlex/configs/text_recognition/PP-OCRv4_mobile_rec.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/ocr_rec_dataset_examples \
    -o CheckDataset.split.enable=True \
    -o CheckDataset.split.train_percent=90 \
    -o CheckDataset.split.val_percent=10
</code></pre></details>

### 4.2 Model Training
Model training can be completed with a single command. Here's an example of training the PP-OCRv4 mobile text recognition model (PP-OCRv4_mobile_rec):

```bash
python main.py -c paddlex/configs/text_recognition/PP-OCRv4_mobile_rec.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/ocr_rec_dataset_examples
```
The steps required are:

* Specify the path to the model's `.yaml` configuration file (here it's `PP-OCRv4_mobile_rec.yaml`,When training other models, you need to specify the corresponding configuration files. The relationship between the model and configuration files can be found in the [PaddleX Model List (CPU/GPU)](../../../support_list/models_list.en.md))
* Specify the mode as model training: `-o Global.mode=train`
* Specify the path to the training dataset: `-o Global.dataset_dir`.
Other related parameters can be set by modifying the `Global` and `Train` fields in the `.yaml` configuration file or adjusted by appending parameters in the command line. For example, to specify training on the first 2 GPUs: `-o Global.device=gpu:0,1`; to set the number of training epochs to 10: `-o Train.epochs_iters=10`. For more modifiable parameters and their detailed explanations, refer to the [PaddleX Common Configuration File Parameters](../../instructions/config_parameters_common.en.md).


<details><summary>👉 <b>More Information (Click to Expand)</b></summary>

<ul>
<li>During model training, PaddleX automatically saves the model weight files, with the default being <code>output</code>. If you need to specify a save path, you can set it through the <code>-o Global.output</code> field in the configuration file.</li>
<li>PaddleX shields you from the concepts of dynamic graph weights and static graph weights. During model training, both dynamic and static graph weights are produced, and static graph weights are selected by default for model inference.</li>
<li>
<p>After completing the model training, all outputs are saved in the specified output directory (default is <code>./output/</code>), typically including:</p>
</li>
<li>
<p><code>train_result.json</code>: Training result record file, recording whether the training task was completed normally, as well as the output weight metrics, related file paths, etc.;</p>
</li>
<li><code>train.log</code>: Training log file, recording changes in model metrics and loss during training;</li>
<li><code>config.yaml</code>: Training configuration file, recording the hyperparameter configuration for this training session;</li>
<li><code>.pdparams</code>, <code>.pdema</code>, <code>.pdopt.pdstate</code>, <code>.pdiparams</code>, <code>.pdmodel</code>: Model weight-related files, including network parameters, optimizer, EMA, static graph network parameters, static graph network structure, etc.;</li>
</ul></details>

## <b>4.3 Model Evaluation</b>
After completing model training, you can evaluate the specified model weights file on the validation set to verify the model's accuracy. Using PaddleX for model evaluation can be done with a single command:

```bash

```bash
python main.py -c paddlex/configs/text_recognition/PP-OCRv4_mobile_rec.yaml \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/ocr_rec_dataset_examples

```
Similar to model training, the following steps are required:

* Specify the `.yaml` configuration file path for the model (here it's `PP-OCRv4_mobile_rec.yaml`)
* Specify the mode as model evaluation: `-o Global.mode=evaluate`
* Specify the path to the validation dataset: `-o Global.dataset_dir`
Other related parameters can be set by modifying the `Global` and `Evaluate` fields in the `.yaml` configuration file. For details, refer to [PaddleX Common Model Configuration File Parameter Description](../../instructions/config_parameters_common.en.md).


<details><summary>👉 <b>More Information (Click to Expand)</b></summary>

<p>When evaluating the model, you need to specify the model weights file path. Each configuration file has a default weight save path. If you need to change it, simply append the command line parameter to set it, such as <code>-o Evaluate.weight_path=./output/best_model/best_model.pdparams</code>.</p>
<p>After completing the model evaluation, an <code>evaluate_result.json</code> file will be produced, which records the evaluation results, specifically, whether the evaluation task was completed successfully and the model's evaluation metrics, including  acc、norm_edit_dis；</p></details>

### <b>4.4 Model Inference and Model Integration</b>
After completing model training and evaluation, you can use the trained model weights for inference prediction or Python integration.

#### 4.4.1 Model Inference
To perform inference prediction via the command line, simply use the following command:

Before running the following code, please download the [demo image](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_rec_001.png) to your local machine.


```bash
python main.py -c paddlex/configs/text_recognition/PP-OCRv4_mobile_rec.yaml \
    -o Global.mode=predict \
    -o Predict.model_dir="./output/best_accuracy/inference" \
    -o Predict.input="general_ocr_rec_001.png"
```
Similar to model training and evaluation, the following steps are required:

* Specify the `.yaml` configuration file path for the model (here it is `PP-OCRv4_mobile_rec.yaml`)
* Specify the mode as model inference prediction: `-o Global.mode=predict`
* Specify the model weights path: `-o Predict.model_dir="./output/best_accuracy/inference"`
* Specify the input data path: `-o Predict.input="..."`
Other related parameters can be set by modifying the `Global` and `Predict` fields in the `.yaml` configuration file. For details, refer to [PaddleX Common Model Configuration File Parameter Description](../../instructions/config_parameters_common.en.md).

#### 4.4.2 Model Integration
Models can be directly integrated into the PaddleX pipelines or into your own projects.

1.<b>Pipeline Integration</b>

The text recognition module can be integrated into PaddleX pipelines such as the [General OCR Pipeline](../../../pipeline_usage/tutorials/ocr_pipelines/OCR.en.md), [General Table Recognition Pipeline](../../../pipeline_usage/tutorials/ocr_pipelines/table_recognition.en.md), and [Document Scene Information Extraction Pipeline v3 (PP-ChatOCRv3)](../../../pipeline_usage/tutorials/information_extraction_pipelines/document_scene_information_extraction.en.md). Simply replace the model path to update the text recognition module of the relevant pipeline.

2.<b>Module Integration</b>

The weights you produce can be directly integrated into the text recognition module. Refer to the [Quick Integration](#iii-quick-integration) Python example code. Simply replace the model with the path to your trained model.
