# 模型推理 Benchmark

PaddleX 支持统计模型推理耗时，需通过环境变量进行设置，具体如下：

* `PADDLE_PDX_INFER_BENCHMARK`：设置为 `True` 时则开启 Benchmark，默认为 `False`；
* `PADDLE_PDX_INFER_BENCHMARK_WARMUP`：设置 warm up，在开始测试前，使用随机数据循环迭代 n 次，默认为 `0`；
* `PADDLE_PDX_INFER_BENCHMARK_DATA_SIZE`： 设置随机数据的尺寸，默认为 `224`；
* `PADDLE_PDX_INFER_BENCHMARK_ITER`：使用随机数据进行 Benchmark 测试的循环次数，仅当输入数据为 `None` 时，将使用随机数据进行测试；
* `PADDLE_PDX_INFER_BENCHMARK_OUTPUT`：用于设置保存的目录，如 `./benchmark`，默认为 `None`，表示不保存 Benchmark 指标；

使用示例如下：

```bash
PADDLE_PDX_INFER_BENCHMARK=True \
PADDLE_PDX_INFER_BENCHMARK_WARMUP=5 \
PADDLE_PDX_INFER_BENCHMARK_DATA_SIZE=320 \
PADDLE_PDX_INFER_BENCHMARK_ITER=10 \
PADDLE_PDX_INFER_BENCHMARK_OUTPUT=./benchmark \
python main.py \
    -c ./paddlex/configs/object_detection/PicoDet-XS.yaml \
    -o Global.mode=predict \
    -o Predict.model_dir=None \
    -o Predict.batch_size=2 \
    -o Predict.input=None
```

在开启 Benchmark 后，将自动打印 benchmark 指标：

```
+----------------+-----------------+-----------------+------------------------+
|   Component    | Total Time (ms) | Number of Calls | Avg Time Per Call (ms) |
+----------------+-----------------+-----------------+------------------------+
|    ReadCmp     |   100.20136833  |        10       |      10.02013683       |
|     Resize     |   17.05980301   |        20       |       0.85299015       |
|   Normalize    |   45.44949532   |        20       |       2.27247477       |
|   ToCHWImage   |    0.03671646   |        20       |       0.00183582       |
|    Copy2GPU    |   12.28785515   |        10       |       1.22878551       |
|     Infer      |   76.59482956   |        10       |       7.65948296       |
|    Copy2CPU    |    0.39863586   |        10       |       0.03986359       |
| DetPostProcess |    0.43916702   |        20       |       0.02195835       |
+----------------+-----------------+-----------------+------------------------+
+-------------+-----------------+---------------------+----------------------------+
|    Stage    | Total Time (ms) | Number of Instances | Avg Time Per Instance (ms) |
+-------------+-----------------+---------------------+----------------------------+
|  PreProcess |   162.74738312  |          20         |         8.13736916         |
|  Inference  |   89.28132057   |          20         |         4.46406603         |
| PostProcess |    0.43916702   |          20         |         0.02195835         |
|   End2End   |    0.27992606   |          20         |         0.01399630         |
|    WarmUp   |    5.37562728   |          5          |         1.07512546         |
+-------------+-----------------+---------------------+----------------------------+
```

在 Benchmark 结果中，会统计该模型全部组件（`Component`）的总耗时（`Total Time`，单位为“毫秒”）、**调用次数**（`Number of Calls`）、**调用**平均执行耗时（`Avg Time Per Call`，单位“毫秒”），以及按预热（`WarmUp`）、预处理（`PreProcess`）、模型推理（`Inference`）、后处理（`PostProcess`）和端到端（`End2End`）进行划分的耗时统计，包括每个阶段的总耗时（`Total Time`，单位为“毫秒”）、**样本数**（`Number of Instances`）和**单样本**平均执行耗时（`Avg Time Per Instance`，单位“毫秒”），同时，上述指标会保存到到本地： `./benchmark/detail.csv` 和 `./benchmark/summary.csv`：

```csv
Component,Total Time (ms),Number of Calls,Avg Time Per Call (ms)
ReadCmp,100.20136833190918,10,10.020136833190918
Resize,17.059803009033203,20,0.8529901504516602
Normalize,45.44949531555176,20,2.272474765777588
ToCHWImage,0.036716461181640625,20,0.0018358230590820312
Copy2GPU,12.28785514831543,10,1.228785514831543
Infer,76.59482955932617,10,7.659482955932617
Copy2CPU,0.3986358642578125,10,0.03986358642578125
DetPostProcess,0.4391670227050781,20,0.021958351135253906
```

```csv
Stage,Total Time (ms),Number of Instances,Avg Time Per Instance (ms)
PreProcess,162.74738311767578,20,8.137369155883789
Inference,89.28132057189941,20,4.464066028594971
PostProcess,0.4391670227050781,20,0.021958351135253906
End2End,0.279926061630249,20,0.013996303081512451
WarmUp,5.375627279281616,5,1.0751254558563232
```
