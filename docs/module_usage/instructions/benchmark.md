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
|    ReadCmp     |   102.39458084  |        10       |      10.23945808       |
|     Resize     |   11.20400429   |        20       |       0.56020021       |
|   Normalize    |   34.11078453   |        20       |       1.70553923       |
|   ToCHWImage   |    0.05555153   |        20       |       0.00277758       |
|    Copy2GPU    |    9.10568237   |        10       |       0.91056824       |
|     Infer      |   98.22225571   |        10       |       9.82222557       |
|    Copy2CPU    |   14.30845261   |        10       |       1.43084526       |
| DetPostProcess |    0.45251846   |        20       |       0.02262592       |
+----------------+-----------------+-----------------+------------------------+
+-------------+-----------------+---------------------+----------------------------+
|    Stage    | Total Time (ms) | Number of Instances | Avg Time Per Instance (ms) |
+-------------+-----------------+---------------------+----------------------------+
|  PreProcess |   147.76492119  |          20         |         7.38824606         |
|  Inference  |   121.63639069  |          20         |         6.08181953         |
| PostProcess |    0.45251846   |          20         |         0.02262592         |
|   End2End   |   294.03519630  |          20         |        14.70175982         |
|    WarmUp   |  7937.82591820  |          5          |       1587.56518364        |
+-------------+-----------------+---------------------+----------------------------+
```

在 Benchmark 结果中，会统计该模型全部组件（`Component`）的总耗时（`Total Time`，单位为“毫秒”）、**调用次数**（`Number of Calls`）、**调用**平均执行耗时（`Avg Time Per Call`，单位“毫秒”），以及按预热（`WarmUp`）、预处理（`PreProcess`）、模型推理（`Inference`）、后处理（`PostProcess`）和端到端（`End2End`）进行划分的耗时统计，包括每个阶段的总耗时（`Total Time`，单位为“毫秒”）、**样本数**（`Number of Instances`）和**单样本**平均执行耗时（`Avg Time Per Instance`，单位“毫秒”），同时，上述指标会保存到到本地： `./benchmark/detail.csv` 和 `./benchmark/summary.csv`：

```csv
Component,Total Time (ms),Number of Calls,Avg Time Per Call (ms)
ReadCmp,0.10199093818664551,10,0.01019909381866455
Resize,0.011309385299682617,20,0.0005654692649841309
Normalize,0.035140275955200195,20,0.0017570137977600097
ToCHWImage,4.744529724121094e-05,20,2.3722648620605467e-06
Copy2GPU,0.00861215591430664,10,0.000861215591430664
Infer,0.820899248123169,10,0.08208992481231689
Copy2CPU,0.006002187728881836,10,0.0006002187728881836
DetPostProcess,0.0004436969757080078,20,2.218484878540039e-05
```

```csv
Stage,Total Time (ms),Number of Instance,Avg Time Per Instance (ms)
PreProcess,0.14848804473876953,20,0.007424402236938477
Inference,0.8355135917663574,20,0.04177567958831787
PostProcess,0.0004436969757080078,20,2.218484878540039e-05
End2End,1.0054960250854492,20,0.05027480125427246
WarmUp,8.869974851608276,5,1.7739949703216553
```
