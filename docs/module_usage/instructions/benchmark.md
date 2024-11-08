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
+----------------+-----------------+------+---------------+
|     Stage      | Total Time (ms) | Nums | Avg Time (ms) |
+----------------+-----------------+------+---------------+
|    ReadCmp     |   185.48870087  |  10  |  18.54887009  |
|     Resize     |   16.95227623   |  30  |   0.56507587  |
|   Normalize    |   41.12100601   |  30  |   1.37070020  |
|   ToCHWImage   |    0.05745888   |  30  |   0.00191530  |
|    Copy2GPU    |   14.58549500   |  10  |   1.45854950  |
|     Infer      |   100.14462471  |  10  |  10.01446247  |
|    Copy2CPU    |    9.54508781   |  10  |   0.95450878  |
| DetPostProcess |    0.56767464   |  30  |   0.01892249  |
+----------------+-----------------+------+---------------+
+-------------+-----------------+------+---------------+
|    Stage    | Total Time (ms) | Nums | Avg Time (ms) |
+-------------+-----------------+------+---------------+
|  PreProcess |   243.61944199  |  30  |   8.12064807  |
|  Inference  |   124.27520752  |  30  |   4.14250692  |
| PostProcess |    0.56767464   |  30  |   0.01892249  |
|   End2End   |   379.70948219  |  30  |  12.65698274  |
|    WarmUp   |  9465.68179131  |  5   | 1893.13635826 |
+-------------+-----------------+------+---------------+
```

在 Benchmark 结果中，会统计该模型全部组件（`Component`）的总耗时（`Total Time`，单位为“毫秒”）、**调用次数**（`Nums`）、**调用**平均执行耗时（`Avg Time`，单位为“毫秒”），以及按预热（`WarmUp`）、预处理（`PreProcess`）、模型推理（`Inference`）、后处理（`PostProcess`）和端到端（`End2End`）进行划分的耗时统计，包括每个阶段的总耗时（`Total Time`，单位为“毫秒”）、**样本数**（`Nums`）和**单样本**平均执行耗时（`Avg Time`，单位为“毫秒”），同时，保存相关指标会到本地 `./benchmark.csv` 文件中：

```csv
Stage,Total Time (ms),Nums,Avg Time (ms)
ReadCmp,0.18548870086669922,10,0.018548870086669923
Resize,0.0169522762298584,30,0.0005650758743286133
Normalize,0.04112100601196289,30,0.001370700200398763
ToCHWImage,5.745887756347656e-05,30,1.915295918782552e-06
Copy2GPU,0.014585494995117188,10,0.0014585494995117188
Infer,0.10014462471008301,10,0.0100144624710083
Copy2CPU,0.009545087814331055,10,0.0009545087814331055
DetPostProcess,0.0005676746368408203,30,1.892248789469401e-05
PreProcess,0.24361944198608398,30,0.0081206480662028
Inference,0.12427520751953125,30,0.0041425069173177086
PostProcess,0.0005676746368408203,30,1.892248789469401e-05
End2End,0.37970948219299316,30,0.012656982739766438
WarmUp,9.465681791305542,5,1.8931363582611085
```
