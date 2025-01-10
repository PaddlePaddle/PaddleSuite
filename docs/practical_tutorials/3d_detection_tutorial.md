---
comments: true
---

# PaddleX 3.0 多模态3D目标检测模型产线———BEVFusion检测教程

PaddleX 提供了丰富的模型产线，模型产线由一个或多个模型组合实现，每个模型产线都能够解决特定的场景任务问题。PaddleX 所提供的模型产线均支持快速体验，如果效果不及预期，也同样支持使用私有数据微调模型，并且 PaddleX 提供了 Python API，方便将产线集成到个人项目中。在使用之前，您首先需要安装 PaddleX， 安装方式请参考[ PaddleX 安装](../installation/installation.md)。此处以一个BEVFusion检测的任务为例子，介绍模型产线工具的使用流程。

## 1. 选择产线

首先，需要根据您的任务场景，选择对应的 PaddleX 产线，此处为BEVFusion检测，需要了解到这个任务属于3D目标检测任务，对应 PaddleX 的3D目标检测产线。如果无法确定任务和产线的对应关系，您可以在 PaddleX 支持的[模型产线列表](../support_list/pipelines_list.md)中了解相关产线的能力介绍。


## 2. 快速体验

暂不支持

## 3. 选择模型

PaddleX 提供了 1 个端到端的3D检测模型，具体可参考 [模型列表](../support_list/models_list.md)。

## 4. 数据准备和校验
### 4.1 数据准备

本教程采用基于 `nuScenes`数据集提取的demo数据集作为示例数据集，数据集下载路径和准备方式如下

```bash
cd /path/to/paddlex
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/nuscenes_demo.tar -P ./dataset
tar -xf ./dataset/nuscenes_demo.tar -C ./dataset/
```

将数据集目录准备如下：
```
dataset
|—— samples
|—— sweeps
|—— nuscenes_infos_train.pkl
|—— nuscenes_infos_val.pkl
```


### 4.2 数据集校验

在对数据集校验时，只需一行命令：

```bash
python main.py -c paddlex/configs/modules/bev_fusion_3D/bevf_pp_2x8_1x_nusc.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/nuscenes_demo
```

执行上述命令后，PaddleX 会对数据集进行校验。命令运行成功后会在 log 中打印出 `Check dataset passed !` 信息，同时相关产出会保存在当前目录的 `./output/check_dataset` 目录下。校验结果文件保存在 `./output/check_dataset_result.json`，校验结果文件具体内容为
```
{
  {
  "done_flag": true,
  "check_pass": true,
  "attributes": {
    "train_mate": [
      {
        "sample_idx": "f9878012c3f6412184c294c13ba4bac3",
        "lidar_path": ".\/data\/nuscenes\/samples\/LIDAR_TOP\/n008-2018-05-21-11-06-59-0400__LIDAR_TOP__1526915243047392.pcd.bin",
        "image_paths": [
          ".\/data\/nuscenes\/samples\/CAM_FRONT_LEFT\/n008-2018-05-21-11-06-59-0400__CAM_FRONT_LEFT__1526915243004917.jpg",
          ".\/data\/nuscenes\/samples\/CAM_FRONT\/n008-2018-05-21-11-06-59-0400__CAM_FRONT__1526915243012465.jpg",
          ".\/data\/nuscenes\/samples\/CAM_FRONT_RIGHT\/n008-2018-05-21-11-06-59-0400__CAM_FRONT_RIGHT__1526915243019956.jpg",
          ".\/data\/nuscenes\/samples\/CAM_BACK_RIGHT\/n008-2018-05-21-11-06-59-0400__CAM_BACK_RIGHT__1526915243027813.jpg",
          ".\/data\/nuscenes\/samples\/CAM_BACK\/n008-2018-05-21-11-06-59-0400__CAM_BACK__1526915243037570.jpg",
          ".\/data\/nuscenes\/samples\/CAM_BACK_LEFT\/n008-2018-05-21-11-06-59-0400__CAM_BACK_LEFT__1526915243047295.jpg"
        ]
      },
    ],
    "val_mate": [
      {
        "sample_idx": "30e55a3ec6184d8cb1944b39ba19d622",
        "lidar_path": ".\/data\/nuscenes\/samples\/LIDAR_TOP\/n015-2018-07-11-11-54-16+0800__LIDAR_TOP__1531281439800013.pcd.bin",
        "image_paths": [
          ".\/data\/nuscenes\/samples\/CAM_FRONT_LEFT\/n015-2018-07-11-11-54-16+0800__CAM_FRONT_LEFT__1531281439754844.jpg",
          ".\/data\/nuscenes\/samples\/CAM_FRONT\/n015-2018-07-11-11-54-16+0800__CAM_FRONT__1531281439762460.jpg",
          ".\/data\/nuscenes\/samples\/CAM_FRONT_RIGHT\/n015-2018-07-11-11-54-16+0800__CAM_FRONT_RIGHT__1531281439770339.jpg",
          ".\/data\/nuscenes\/samples\/CAM_BACK_RIGHT\/n015-2018-07-11-11-54-16+0800__CAM_BACK_RIGHT__1531281439777893.jpg",
          ".\/data\/nuscenes\/samples\/CAM_BACK\/n015-2018-07-11-11-54-16+0800__CAM_BACK__1531281439787525.jpg",
          ".\/data\/nuscenes\/samples\/CAM_BACK_LEFT\/n015-2018-07-11-11-54-16+0800__CAM_BACK_LEFT__1531281439797423.jpg"
        ]
      },
    ]
  },
  "analysis": {},
  "dataset_path": "\/workspace\/bevfusion\/Paddle3D\/data\/nuscenes",
  "show_type": "path for images and lidar",
  "dataset_type": "NuscenesMMDataset"
}
}
```
上述校验结果中，check_pass 为 True 表示数据集格式符合要求。

<b>注</b>：只有通过数据校验的数据才可以训练和评估。


## 5. 模型训练和评估
### 5.1 模型训练

在训练之前，请确保您已经对数据集进行了校验。

完成 PaddleX 模型的训练，只需如下一条命令：

```bash
python main.py -c paddlex/configs/modules/bev_fusion_3D/bevf_pp_2x8_1x_nusc.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/nuscenes_demo \
```

在 PaddleX 中模型训练支持：修改训练超参数、单机单卡/多卡训练等功能，只需修改配置文件或追加命令行参数。

PaddleX 中每个模型都提供了模型开发的配置文件，用于设置相关参数。模型训练相关的参数可以通过修改配置文件中 `Train` 下的字段进行设置，配置文件中部分参数的示例说明如下：

* `Global`：
    * `mode`：模式，支持数据校验（`check_dataset`）、模型训练（`train`）、模型评估（`evaluate`）；
    * `device`：训练设备，可选`cpu`、`gpu`，除 cpu 外，多卡训练可指定卡号，如：`gpu:0,1,2,3,4,5,6,7`；
* `Train`：训练超参数设置；
    * `epochs`：训练轮次数设置；
    * `learning_rate`：训练学习率设置；
    * `batch_size`：训练batch_size设置；

更多超参数介绍，请参考 [PaddleX 通用模型配置文件参数说明](../module_usage/instructions/config_parameters_common.md)。

<b>注：</b>
- 以上参数可以通过追加令行参数的形式进行设置，如指定模式为模型训练：`-o Global.mode=train`；指定前 2 卡 gpu 训练：`-o Global.device=gpu:0,1`；设置训练轮次数为 10：`-o Train.epochs=10`。
- 模型训练过程中，PaddleX 会自动保存模型权重文件，默认为`output`，如需指定保存路径，可通过配置文件中 `-o Global.output` 字段
- PaddleX 对您屏蔽了动态图权重和静态图权重的概念。在模型训练的过程中，会同时产出动态图和静态图的权重，在模型推理时，默认选择静态图权重推理。

<b>训练产出解释:</b>

在完成模型训练后，所有产出保存在指定的输出目录（默认为`./output/`）下，通常有以下产出：

* train_result.json：训练结果记录文件，记录了训练任务是否正常完成，以及产出的权重指标、相关文件路径等；
* train.log：训练日志文件，记录了训练过程中的模型指标变化、loss 变化等；
* config.yaml：训练配置文件，记录了本次训练的超参数的配置；
* .pdparams、.pdema、.pdopt.pdstate、.pdiparams、.pdmodel：模型权重相关文件，包括网络参数、优化器、EMA、静态图网络参数、静态图网络结构等；

### 5.2 模型评估

在完成模型训练后，可以对指定的模型权重文件在验证集上进行评估，验证模型精度。使用 PaddleX 进行模型评估，只需一行命令：

```bash
python main.py -c paddlex/configs/modules/bev_fusion_3D/bevf_pp_2x8_1x_nusc.yaml \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/nuscenes_demo \
    -o Evaluate.weight_path=output_pdparams_path
```

与模型训练类似，模型评估支持修改配置文件或追加命令行参数的方式设置。


## 6. 产线测试

暂不支持

## 7. 开发集成/部署

暂不支持
