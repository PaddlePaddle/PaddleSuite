# 模型推理 Benchmark

PaddleX 支持统计模型推理耗时，需通过环境变量进行设置，具体如下：

* `PADDLE_PDX_INFER_BENCHMARK`：设置为 `True` 时则开启 Benchmark，默认为 `False`；
* `PADDLE_PDX_INFER_BENCHMARK_WARMUP`：设置 warm up，在开始测试前，使用随机数据循环迭代 n 次，默认为 `0`；
* `PADDLE_PDX_INFER_BENCHMARK_DATA_SIZE`： 设置随机数据的尺寸，默认为 `224`；
* `PADDLE_PDX_INFER_BENCHMARK_ITER`：使用随机数据进行 Benchmark 测试的循环次数，仅当输入数据为 `None` 时，将使用随机数据进行测试；
* `PADDLE_PDX_INFER_BENCHMARK_OUTPUT`：用于设置保存本次 benchmark 指标到 `txt` 文件，如 `./benchmark.txt`，默认为 `None`，表示不保存 Benchmark 指标；

使用示例如下：

```bash
PADDLE_PDX_INFER_BENCHMARK=True \
PADDLE_PDX_INFER_BENCHMARK_WARMUP=5 \
PADDLE_PDX_INFER_BENCHMARK_DATA_SIZE=320 \
PADDLE_PDX_INFER_BENCHMARK_ITER=10 \
PADDLE_PDX_INFER_BENCHMARK_OUTPUT=./benchmark.txt \
python main.py \
    -c ./paddlex/configs/object_detection/PicoDet-XS.yaml \
    -o Global.mode=predict \
    -o Predict.model_dir=None \
    -o Predict.input=None
```

在开启 Benchmark 后，将自动打印 benchmark 指标：

```
+-------------------+--------+------------------+
|     Component     | Counts | Average Time(ms) |
+-------------------+--------+------------------+
|      ReadCmp      |   10   |    7.86035061    |
|       Resize      |   10   |    1.38545036    |
|     Normalize     |   10   |    3.77433300    |
|     ToCHWImage    |   10   |    0.00545979    |
| ImageDetPredictor |   10   |   14.97282982    |
|   DetPostProcess  |   10   |    0.06134510    |
|  ***************  | ****** | ***************  |
|     PreProcess    |   \    |   13.02559376    |
|     Inference     |   \    |   14.97282982    |
|    PostProcess    |   \    |    0.06134510    |
+-------------------+--------+------------------+
```

在 Benchmark 结果中，会统计该模型全部组件（`Component`）的平均执行耗时（`Average Time`，单位为“毫秒”）和调用次数（`Counts`），以及按预处理（`PreProcess`）、模型推理（`Inference`）和后处理（`PostProcess`）汇总得到的执行耗时，同时，保存相关指标会到本地 `./benchmark.txt` 文件中：

```
Component, Counts, Average Time(ms)
ReadCmp, 10, 7.860350608825682706
Resize, 10, 1.385450363159179688
Normalize, 10, 3.774333000183105469
ToCHWImage, 10, 0.005459785461425781
ImageDetPredictor, 10, 14.972829818725585938
DetPostProcess, 10, 0.061345100402832031
***************, ***, ***************
PreProcess, \, 13.025593757629394531
Inference, \, 14.972829818725585938
PostProcess, \, 0.061345100402832031
```
