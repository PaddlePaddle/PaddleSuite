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
+-------------------+-------------+------------------------+
|     Component     | Call Counts | Avg Time Per Call (ms) |
+-------------------+-------------+------------------------+
|      ReadCmp      |     1000    |      19.22814894       |
|       Resize      |     1000    |       2.52388239       |
|     Normalize     |     1000    |       1.33547258       |
|     ToCHWImage    |     1000    |       0.00310326       |
| ImageDetPredictor |     1000    |       6.83180261       |
|   DetPostProcess  |     1000    |       0.03265357       |
+-------------------+-------------+------------------------+
+-------------+------------------+----------------------------+
|    Stage    | Num of Instances | Avg Time Per Instance (ms) |
+-------------+------------------+----------------------------+
|  PreProcess |       1000       |        23.09060717         |
|  Inference  |       1000       |         6.83180261         |
| PostProcess |       1000       |         0.03265357         |
|   End2End   |       1000       |        30.48534989         |
+-------------+------------------+----------------------------+
```

在 Benchmark 结果中，会统计该模型全部组件（`Component`）的平均执行耗时（`Avg Time Per Call`，单位为“毫秒”）和调用次数（`Call Counts`），以及按预处理（`PreProcess`）、模型推理（`Inference`）、后处理（`PostProcess`）和端到端（`End2End`）汇总得到的单样本平均耗时（`Avg Time Per Instance`，单位为“毫秒”），同时，保存相关指标会到本地 `./benchmark.txt` 文件中：

```
Component, Call Counts, Avg Time Per Call (ms)
ReadCmp, 1000, 19.329239845275878906
Resize, 1000, 2.562829017639160156
Normalize, 1000, 1.369090795516967773
ToCHWImage, 1000, 0.003165960311889648
ImageDetPredictor, 1000, 7.323185205459594727
DetPostProcess, 1000, 0.033131122589111328
****************************************************************************************************
Stage, Num of Instances, Avg Time Per Instance (ms)
PreProcess, 1000, 23.264325618743896484
Inference, 1000, 7.323185205459594727
PostProcess, 1000, 0.033131122589111328
End2End, 1000, 31.181738615036010742
```
