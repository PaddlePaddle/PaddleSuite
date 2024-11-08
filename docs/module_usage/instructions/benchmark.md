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
+-------------------+-----------------+------+---------------+
|       Stage       | Total Time (ms) | Nums | Avg Time (ms) |
+-------------------+-----------------+------+---------------+
|      ReadCmp      |   49.95107651   |  10  |   4.99510765  |
|       Resize      |    8.48054886   |  10  |   0.84805489  |
|     Normalize     |   23.08964729   |  10  |   2.30896473  |
|     ToCHWImage    |    0.02717972   |  10  |   0.00271797  |
| ImageDetPredictor |   75.94108582   |  10  |   7.59410858  |
|   DetPostProcess  |    0.26535988   |  10  |   0.02653599  |
+-------------------+-----------------+------+---------------+
+-------------+-----------------+------+---------------+
|    Stage    | Total Time (ms) | Nums | Avg Time (ms) |
+-------------+-----------------+------+---------------+
|  PreProcess |   81.54845238   |  10  |   8.15484524  |
|  Inference  |   75.94108582   |  10  |   7.59410858  |
| PostProcess |    0.26535988   |  10  |   0.02653599  |
|   End2End   |   161.07797623  |  10  |  16.10779762  |
|    WarmUp   |  5496.41847610  |  5   | 1099.28369522 |
+-------------+-----------------+------+---------------+
```

在 Benchmark 结果中，会统计该模型全部组件（`Component`）的总耗时（`Total Time`，单位为“毫秒”）、调用次数（`Nums`）、调用平均执行耗时（`Avg Time`，单位为“毫秒”），以及按预热（`WarmUp`）、预处理（`PreProcess`）、模型推理（`Inference`）、后处理（`PostProcess`）和端到端（`End2End`）进行划分的耗时统计，包括每个阶段的总耗时（`Total Time`，单位为“毫秒”）、样本数（`Nums`）和单样本平均执行耗时（`Avg Time`，单位为“毫秒”），同时，保存相关指标会到本地 `./benchmark.csv` 文件中：

```csv
Stage,Total Time (ms),Nums,Avg Time (ms)
ReadCmp,0.04995107650756836,10,0.004995107650756836
Resize,0.008480548858642578,10,0.0008480548858642578
Normalize,0.02308964729309082,10,0.002308964729309082
ToCHWImage,2.7179718017578125e-05,10,2.7179718017578126e-06
ImageDetPredictor,0.07594108581542969,10,0.007594108581542969
DetPostProcess,0.00026535987854003906,10,2.6535987854003906e-05
PreProcess,0.08154845237731934,10,0.008154845237731934
Inference,0.07594108581542969,10,0.007594108581542969
PostProcess,0.00026535987854003906,10,2.6535987854003906e-05
End2End,0.16107797622680664,10,0.016107797622680664
WarmUp,5.496418476104736,5,1.0992836952209473
```
