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
|    ReadCmp     |   99.60412979   |        10       |       9.96041298       |
|     Resize     |   17.01641083   |        20       |       0.85082054       |
|   Normalize    |   44.61312294   |        20       |       2.23065615       |
|   ToCHWImage   |    0.03385544   |        20       |       0.00169277       |
|    Copy2GPU    |   13.46874237   |        10       |       1.34687424       |
|     Infer      |   71.31743431   |        10       |       7.13174343       |
|    Copy2CPU    |    0.39076805   |        10       |       0.03907681       |
| DetPostProcess |    0.36168098   |        20       |       0.01808405       |
+----------------+-----------------+-----------------+------------------------+
+-------------+-----------------+---------------------+----------------------------+
|    Stage    | Total Time (ms) | Number of Instances | Avg Time Per Instance (ms) |
+-------------+-----------------+---------------------+----------------------------+
|  PreProcess |   161.26751900  |          20         |         8.06337595         |
|  Inference  |   85.17694473   |          20         |         4.25884724         |
| PostProcess |    0.36168098   |          20         |         0.01808405         |
|   End2End   |   256.90770149  |          20         |        12.84538507         |
|    WarmUp   |  5412.37807274  |          10         |        541.23780727        |
+-------------+-----------------+---------------------+----------------------------+
```

在 Benchmark 结果中，会统计该模型全部组件（`Component`）的总耗时（`Total Time`，单位为“毫秒”）、**调用次数**（`Number of Calls`）、**调用**平均执行耗时（`Avg Time Per Call`，单位“毫秒”），以及按预热（`WarmUp`）、预处理（`PreProcess`）、模型推理（`Inference`）、后处理（`PostProcess`）和端到端（`End2End`）进行划分的耗时统计，包括每个阶段的总耗时（`Total Time`，单位为“毫秒”）、**样本数**（`Number of Instances`）和**单样本**平均执行耗时（`Avg Time Per Instance`，单位“毫秒”），同时，上述指标会保存到到本地： `./benchmark/detail.csv` 和 `./benchmark/summary.csv`：

```csv
Component,Total Time (ms),Number of Calls,Avg Time Per Call (ms)
ReadCmp,99.60412979125977,10,9.960412979125977
Resize,17.01641082763672,20,0.8508205413818359
Normalize,44.61312294006348,20,2.230656147003174
ToCHWImage,0.033855438232421875,20,0.0016927719116210938
Copy2GPU,13.468742370605469,10,1.3468742370605469
Infer,71.31743431091309,10,7.131743431091309
Copy2CPU,0.39076805114746094,10,0.039076805114746094
DetPostProcess,0.3616809844970703,20,0.018084049224853516
```

```csv
Stage,Total Time (ms),Number of Instances,Avg Time Per Instance (ms)
PreProcess,161.26751899719238,20,8.06337594985962
Inference,85.17694473266602,20,4.258847236633301
PostProcess,0.3616809844970703,20,0.018084049224853516
End2End,256.90770149230957,20,12.845385074615479
WarmUp,5412.3780727386475,10,541.2378072738647
```
