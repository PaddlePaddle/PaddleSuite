English | [简体中文](face_feature.md)

# Face Feature Module Usage Tutorial

## I. Overview
Face feature models typically take standardized face images processed through detection, extraction, and keypoint correction as input. These models extract highly discriminative facial features from these images for subsequent modules, such as face matching and verification tasks.

## II. Supported Model List

<details>
   <summary> 👉Details of Model List</summary>

| Model           | Output Feature Dimension | Acc (%)<br>AgeDB-30/CFP-FP/LFW | GPU Inference Time (ms) | CPU Inference Time | Model Size (M) | Description                                  |
|---------------|--------|-------------------------------|--------------|---------|------------|-------------------------------------|
| MobileFaceNet | 128    | 96.28/96.71/99.58             |              |         | 4.1        | Face feature model trained on MobileFaceNet with MS1Mv3 dataset |
| ResNet50_face    | 512    | 98.12/98.56/99.77             |              |         | 87.2       | Face feature model trained on ResNet50 with MS1Mv3 dataset      |

Note: The above accuracy metrics are Accuracy scores measured on the AgeDB-30, CFP-FP, and LFW datasets, respectively. All model GPU inference times are based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speeds are based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.
</details>

## III. Quick Integration
> ❗ Before quick integration, please install the PaddleX wheel package. For details, refer to the [PaddleX Local Installation Tutorial](../../../installation/installation_en.md)

After installing the whl package, a few lines of code can complete the inference of the face feature module. You can switch models under this module freely, and you can also integrate the model inference of the face feature module into your project. Before running the following code, please download the [example image](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/face_recognition_001.jpg) to your local machine.

```python
from paddlex import create_model

model_name = "MobileFaceNet"

model = create_model(model_name)
output = model.predict("face_recognition_001.jpg", batch_size=1)

for res in output:
    res.print(json_format=False)
    res.save_to_json("./output/res.json")
```

For more information on using the PaddleX single-model inference API, refer to the [PaddleX Single Model Python Script Usage Instructions](../../instructions/model_python_API_en.md).

## IV. Custom Development
If you aim for higher accuracy with existing models, you can leverage PaddleX's custom development capabilities to develop better face feature models. Before developing face feature models with PaddleX, ensure you have installed the PaddleX PaddleClas plugin. The installation process can be found in the [PaddleX Local Installation Tutorial](../../../installation/installation_en.md)

### 4.1 Data Preparation
Before model training, you need to prepare the dataset for the corresponding task module. PaddleX provides data validation functionality for each module, and **only data that passes validation can be used for model training**. Additionally, PaddleX provides demo datasets for each module, allowing you to complete subsequent development based on the official demo data. If you wish to use a private dataset for subsequent model training, the training dataset for the face feature module is organized in a general image classification dataset format. You can refer to the [PaddleX Image Classification Task Module Data Annotation Tutorial](../../../data_annotations/cv_modules/image_classification_en.md). If you wish to use a private dataset for subsequent model evaluation, note that the validation dataset format for the face feature module differs from the training dataset format. Please refer to [Section 4.1.4 Data Organization Face Feature Module](#414-Data-Organization-for-Face-Feature-Module)

#### 4.1.1 Demo Data Download
You can use the following commands to download the demo dataset to a specified folder:

```bash
cd /path/to/paddlex
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/face_rec_examples.tar -P ./dataset
tar -xf ./dataset/face_rec_examples.tar -C ./dataset/
```
#### 4.1.2 Data Validation
A single command can complete data validation:

```bash
python main.py -c paddlex/configs/face_recognition/MobileFaceNet.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/face_rec_examples
```

After executing the above command, PaddleX will validate the dataset and collect its basic information. Upon successful execution, the log will print the message `Check dataset passed !`. The validation result file will be saved in `./output/check_dataset_result.json`, and related outputs will be saved in the `./output/check_dataset` directory of the current directory. The output directory includes visualized example images and histograms of sample distributions.

<details>
  <summary>👉 <b>Validation Result Details (Click to Expand)</b></summary>

The specific content of the validation result file is:

```bash
{
  "done_flag": true,
  "check_pass": true,
  "attributes": {
    "train_label_file": "../../dataset/face_rec_examples/train/label.txt",
    "train_num_classes": 995,
    "train_samples": 1000,
    "train_sample_paths": [
      "check_dataset/demo_img/01378592.jpg",
      "check_dataset/demo_img/04331410.jpg",
      "check_dataset/demo_img/03485713.jpg",
      "check_dataset/demo_img/02382123.jpg",
      "check_dataset/demo_img/01722397.jpg",
      "check_dataset/demo_img/02682349.jpg",
      "check_dataset/demo_img/00272794.jpg",
      "check_dataset/demo_img/03151987.jpg",
      "check_dataset/demo_img/01725764.jpg",
      "check_dataset/demo_img/02580369.jpg"
    ],
    "val_label_file": "../../dataset/face_rec_examples/val/pair_label.txt",
    "val_num_classes": 2,
    "val_samples": 500
  },
  "analysis": {},
  "dataset_path": "./dataset/face_rec_examples",
  "show_type": "image",
  "dataset_type": "ClsDataset"
}
```

The verification results mentioned above indicate that `check_pass` being `True` means the dataset format meets the requirements. Details of other indicators are as follows:

* `attributes.train_num_classes`: The number of classes in this training dataset is 995;
* `attributes.val_num_classes`: The number of classes in this validation dataset is 2;
* `attributes.train_samples`: The number of training samples in this dataset is 1000;
* `attributes.val_samples`: The number of validation samples in this dataset is 500;
* `attributes.train_sample_paths`: The list of relative paths to the visualization images of training samples in this dataset;

</details>

#### 4.1.3 Dataset Format Conversion / Dataset Splitting (Optional)
After completing the data validation, you can convert the dataset format and re-split the training/validation ratio by **modifying the configuration file** or **adding hyperparameters**.

<details>
  <summary>👉 <b>Details on Format Conversion / Dataset Splitting (Click to Expand)</b></summary>

The face feature module does not support data format conversion or dataset splitting.

</details>

#### 4.1.4 Data Organization for Face Feature Module

The format of the validation dataset for the face feature module differs from the training dataset. If you need to evaluate model accuracy on private data, please organize your dataset as follows:

```bash
face_rec_dataroot      # Root directory of the dataset, the directory name can be changed
├── train              # Directory for saving the training dataset, the directory name cannot be changed
   ├── images          # Directory for saving images, the directory name can be changed but should correspond to the content in label.txt
      ├── xxx.jpg      # Face image file
      ├── xxx.jpg      # Face image file
      ...
   └──label.txt       # Training set annotation file, the file name cannot be changed. Each line gives the relative path of the image to `train` and the face image class (face identity) id, separated by a space. Example content: images/image_06765.jpg 0
├── val                # Directory for saving the validation dataset, the directory name cannot be changed
   ├── images          # Directory for saving images, the directory name can be changed but should correspond to the content in pair_label.txt
      ├── xxx.jpg      # Face image file 
      ├── xxx.jpg      # Face image file 
      ...
   └── pair_label.txt  # Validation dataset annotation file, the file name cannot be changed. Each line gives the paths of two images to be compared and a 0 or 1 label indicating whether the pair of images belong to the same person, separated by spaces.
```

Example content of the validation set annotation file `pair_label.txt`:

```bash
# Face image 1.jpg Face image 2.jpg Label (0 indicates the two face images do not belong to the same person, 1 indicates they do)
images/Angela_Merkel_0001.jpg images/Angela_Merkel_0002.jpg 1
images/Bruce_Gebhardt_0001.jpg images/Masao_Azuma_0001.jpg 0
images/Francis_Ford_Coppola_0001.jpg images/Francis_Ford_Coppola_0002.jpg 1
images/Jason_Kidd_0006.jpg images/Jason_Kidd_0008.jpg 1
images/Miyako_Miyazaki_0002.jpg images/Munir_Akram_0002.jpg 0
```

### 4.2 Model Training
Model training can be completed with a single command. Here is an example of training MobileFaceNet:

```bash
python main.py -c paddlex/configs/face_recognition/MobileFaceNet.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/face_rec_examples
```
The steps required are:

* Specify the path to the `.yaml` configuration file for the model (here it is `MobileFaceNet.yaml`)
* Specify the mode as model training: `-o Global.mode=train`
* Specify the path to the training dataset: `-o Global.dataset_dir`
Other related parameters can be set by modifying the `Global` and `Train` fields in the `.yaml` configuration file or by appending parameters in the command line. For example, to specify the first two GPUs for training: `-o Global.device=gpu:0,1`; to set the number of training epochs to 10: `-o Train.epochs_iters=10`. For more modifiable parameters and their detailed explanations, refer to the configuration file instructions for the corresponding task module [PaddleX Common Configuration Parameters for Model Tasks](../../instructions/config_parameters_common_en.md).

<details>
  <summary>👉 <b>More Details (Click to Expand)</b></summary>

* During model training, PaddleX automatically saves model weight files, defaulting to `output`. To specify a save path, use the `-o Global.output` field in the configuration file.
* PaddleX shields you from the concepts of dynamic graph weights and static graph weights. During model training, both dynamic and static graph weights are produced, and static graph weights are selected by default for model inference.
* When training other models, specify the corresponding configuration file. The correspondence between models and configuration files can be found in the [PaddleX Model List (CPU/GPU)](../../../support_list/models_list_en.md).
After completing model training, all outputs are saved in the specified output directory (default is `./output/`). Typically, the following outputs are included:
* `train_result.json`: A file that records the training results, indicating whether the training task was successfully completed, and includes metrics, paths to related files, etc.
* `train.log`: A log file that records changes in model metrics, loss variations, and other details during the training process.
* `config.yaml`: A configuration file that logs the hyperparameter settings for the current training session.
* `.pdparams`, `.pdema`, `.pdopt.pdstate`, `.pdiparams`, `.pdmodel`: Files related to model weights, including network parameters, optimizer, EMA (Exponential Moving Average), static graph network parameters, and static graph network structure.
<details>

### **4.3 Model Evaluation**
After completing model training, you can evaluate the specified model weight file on the validation set to verify the model's accuracy. Using PaddleX for model evaluation, you can complete the evaluation with a single command:

```bash
python main.py -c paddlex/configs/face_detection/MobileFaceNet.yaml \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/face_rec_examples
```
Similar to model training, the process involves the following steps:

* Specify the path to the `.yaml` configuration file for the model（here it's `MobileFaceNet.yaml`）
* Set the mode to model evaluation: `-o Global.mode=evaluate`
* Specify the path to the validation dataset: `-o Global.dataset_dir`
Other related parameters can be configured by modifying the fields under `Global` and `Evaluate` in the `.yaml` configuration file. For detailed information, please refer to [PaddleX Common Configuration Parameters for Models](../../instructions/config_parameters_common_en.md)。

<details>
  <summary>👉 <b>More Details (Click to Expand)</b></summary>

During model evaluation, the path to the model weights file needs to be specified. Each configuration file has a default weight save path built in. If you need to change it, you can set it by appending a command line parameter, such as `-o Evaluate.weight_path="./output/best_model/best_model/model.pdparams"`.

After completing the model evaluation, an `evaluate_result.json` file will be produced, which records the evaluation results. Specifically, it records whether the evaluation task was completed normally and the model's evaluation metrics, including Accuracy.

</details>

### **4.4 Model Inference**
After completing model training and evaluation, you can use the trained model weights for inference predictions. In PaddleX, model inference predictions can be implemented through two methods: command line and wheel package.

#### 4.4.1 Model Inference
* To perform inference predictions through the command line, you only need the following command. Before running the following code, please download the [example image](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/face_recognition_001.jpg) to your local machine.
```bash
python main.py -c paddlex/configs/face_recognition/MobileFaceNet.yaml \
    -o Global.mode=predict \
    -o Predict.model_dir="./output/best_model/inference" \
    -o Predict.input="face_recognition_001.jpg"
```
Similar to model training and evaluation, the following steps are required:

* Specify the path to the model's `.yaml` configuration file (here it is `MobileFaceNet.yaml`)
* Specify the mode as model inference prediction: `-o Global.mode=predict`
* Specify the path to the model weights: `-o Predict.model_dir="./output/best_model/inference"`
* Specify the path to the input data: `-o Predict.input="..."`
Other related parameters can be set by modifying the fields under `Global` and `Predict` in the `.yaml` configuration file. For details, please refer to [PaddleX Common Model Configuration File Parameter Description](../../instructions/config_parameters_common.md).

#### 4.4.2 Model Integration
The model can be directly integrated into the PaddleX pipeline or into your own project.

1. **Pipeline Integration**

The face feature module can be integrated into the PaddleX pipeline for [**Face Recognition**](../../../pipeline_usage/tutorials/face_recognition_pipelines/face_recognition_en.md). You only need to replace the model path to update the face feature module of the relevant pipeline. In pipeline integration, you can use high-performance deployment and service-oriented deployment to deploy the model you obtained.

2. **Module Integration**

The weights you produced can be directly integrated into the face feature module. You can refer to the Python example code in [Quick Integration](#III.-Quick-Integration) and only need to replace the model with the path to the model you trained.
