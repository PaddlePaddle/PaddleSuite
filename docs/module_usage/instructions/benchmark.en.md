# Model Inference Benchmark

PaddleX support to benchmark model inference. Just set the related flags:

* `PADDLE_PDX_INFER_BENCHMARK`: `True` means enable benchmark. `False` by default;
* `PADDLE_PDX_INFER_BENCHMARK_WARMUP`: Number of warmup. Using random data to infer before testing benchmark if `input` is set to `None`. `0` by default;
* `PADDLE_PDX_INFER_BENCHMARK_DATA_SIZE`: The size of randomly generated data. Valid only when `input` is set to `None`. `224` by default;
* `PADDLE_PDX_INFER_BENCHMARK_ITER`: Number of benchmark testing using random data. Valid only when `input` is set to `None`. `10` by default;
* `PADDLE_PDX_INFER_BENCHMARK_OUTPUT`: The directory to save benchmark result. `None` by default, that means not save.

The example is as follows：

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

The benchmark infomation would be print:

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

The first table show the benchmark infomation by each component(`Component`), include `Total Time` (unit is "ms"), `Number of Calls` and `Avg Time Per Call`  (unit is "ms"). `Avg Time Per Call` is `Total Time` devide by `Number of Calls`. It should be noted that the `Number of Calls` is the number of times the component has been called.

And the second table show the benchmark infomation by different stages: `WarmUp`, `PreProcess`, `Inference`, `PostProcess` and `End2End`. Different from the first table, `Number of Instances` is the number of instances (samples), not the number of calls.

Meanwhile, the benchmark infomation would be saved to local files (`detail.csv` and `summary.csv`) if you set `PADDLE_PDX_INFER_BENCHMARK_OUTPUT`:

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
