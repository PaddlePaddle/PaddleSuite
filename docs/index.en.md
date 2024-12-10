---
comments: true
typora-copy-images-to: images
hide:
  - navigation
  - toc
---

<p align="center">
  <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/logo.png" width="735" height ="200" alt="PaddleX" align="middle" />
</p>

<p align="center">
    <a href=""><img src="https://img.shields.io/badge/License-Apache%202-red.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/Python-3.8%2C%203.9%2C%203.10-blue.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/OS-Linux%2C%20Windows%2C%20Mac-orange.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/Hardware-CPU%2C%20GPU%2C%20XPU%2C%20NPU%2C%20MLU%2C%20DCU-yellow.svg"></a>
</p>



## 🔍 Introduction

PaddleX 3.0 is a low-code development tool for AI models built on the PaddlePaddle framework. It integrates numerous<b>ready-to-use pre-trained models</b>, enabling<b>full-process development</b>from model training to inference, supporting<b>a variety of mainstream hardware</b> both domestic and international, and aiding AI developers in industrial practice.

<style>
        .centered-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        .centered-table th, .centered-table td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: center;
        }
        .centered-table th {
            background-color: #f2f2f2;
            font-weight: bold;
        }
        .centered-table img {
            max-width: 100px;
            height: auto;
        }
        .img-table {
            width: 100%;
            margin: 0 auto;
            border-collapse: collapse;
            text-align: center;
        }
        .img-table th, .centered-table td {
            padding: 10px;
        }
        .img-table img {
            height: 126px;
            width: 180px;
            object-fit: cover;
        }
</style>

<table class="img-table">
        <tr>
            <th><a href="https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/cv_pipelines/image_classification.html"><strong>Image Classification</strong></a></th>
            <th><a href="https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/cv_pipelines/image_multi_label_classification.html"><strong>Multi-label Image Classification</strong></a></th>
            <th><a href="https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/cv_pipelines/object_detection.html"><strong>Object Detection</strong></a></th>
            <th><a href="https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/cv_pipelines/instance_segmentation.html"><strong>Instance Segmentation</strong></a></th>
        </tr>
        <tr>
            <td><img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/b302cd7e-e027-4ea6-86d0-8a4dd6d61f39"></td>
            <td><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/multilabel_cls.png"></td>
            <td><img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/099e2b00-0bbe-4b20-9c5a-96b69e473bd2"></td>
            <td><img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/09f683b4-27df-4c24-b8a7-84da20fdd182"></td>
        </tr>
        <tr>
            <th><a href="https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/cv_pipelines/semantic_segmentation.html"><strong>Semantic Segmentation</strong></a></th>
            <th><a href="https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/cv_pipelines/image_anomaly_detection.html"><strong>Image Anomaly Detection</strong></a></th>
            <th><a href="https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/ocr_pipelines/OCR.html"><strong>OCR</strong></a></th>
            <th><a href="https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/ocr_pipelines/table_recognition.html"><strong>Table Recognition</strong></a></th>
        </tr>
        <tr>
            <td><img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/02637f8c-f248-415b-89ab-1276505f198c"></td>
            <td><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/image_anomaly_detection.png"></td>
            <td><img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/1ef48536-48d4-484b-a6fb-0d6631ba2386"></td>
            <td><img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/1e798e05-dee7-4b41-9cc4-6708b6014efa"></td>
        </tr>
        <tr>
            <th><a href="https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/information_extraction_pipelines/document_scene_information_extraction.html"><strong>PP-ChatOCRv3-doc</strong></a></th>
            <th><a href="https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/time_series_pipelines/time_series_forecasting.html"><strong>Time Series Forecasting</strong></a></th>
            <th><a href="https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/time_series_pipelines/time_series_anomaly_detection.html"><strong>Time Series Anomaly Detection</strong></a></th>
            <th><a href="https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/time_series_pipelines/time_series_classification.html"><strong>Time Series Classification</strong></a></th>
        </tr>
        <tr>
            <td><img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/e3d97f4e-ab46-411c-8155-494c61492b0a"></td>
            <td><img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/6e897bf6-35fe-45e6-a040-e9a1a20cfdf2"></td>
            <td><img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/c54c66cc-da4f-4631-877b-43b0fbb192a6"></td>
            <td><img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/0ce925b2-3776-4dde-8ce0-5156d5a2476e"></td>
        </tr>
    </table>

## 🛠️ Installation

!!! warning
    Please ensure you have a basic <b>Python runtime environment</b> before installing PaddleX (Note: Currently supports Python 3.8 to Python 3.10, with more Python versions being adapted).

### Installing PaddlePaddle

=== "CPU"
    ```bash
    python -m pip install paddlepaddle==3.0.0b2 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
    ```
=== "CUDA 11.8"
    ```bash
    python -m pip install paddlepaddle-gpu==3.0.0b2 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
    ```
=== "CUDA 12.3"
    ```bash
    python -m pip install paddlepaddle-gpu==3.0.0b2 -i https://www.paddlepaddle.org.cn/packages/stable/cu123/
    ```

> ❗ For more PaddlePaddle Wheel versions, please refer to the [PaddlePaddle official website](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation./docs/zh/install/pip/linux-pip.html).


### Installing PaddleX

```bash
pip install paddlex==3.0.0b2
```

> ❗ For more installation methods, please refer to the [PaddleX Installation Guide](https://paddlepaddle.github.io/PaddleX/latest/installation/installation.html)

## 💻 Command Line Usage

A single command can quickly experience the production line effect, with a unified command line format as follows:

```bash
paddlex --pipeline [production line name] --input [input image] --device [running device]
```

You only need to specify three parameters:

* `pipeline`: The name of the production line
* `input`: The local path or URL of the input file to be processed (e.g., an image)
* `device`: The GPU number used (for example, `gpu:0` indicates using the 0th GPU), or you can choose to use CPU (`cpu`)


!!! example "OCR-related CLI"

    === "OCR"

        ```bash
        paddlex --pipeline OCR --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png --device gpu:0
        ```

        ??? question "What's the result"
            === "output"
                ```bash
                'input_path': '/root/.paddlex/predict_input/general_ocr_002.png',
                'dt_polys': [array([[161,  27],
                    [353,  22],
                    [354,  69],
                    [162,  74]], dtype=int16), array([[426,  26],
                    [657,  21],
                    [657,  58],
                    [426,  62]], dtype=int16), array([[702,  18],
                    [822,  13],
                    [824,  57],
                    [704,  62]], dtype=int16), array([[341, 106],
                    [405, 106],
                    [405, 128],
                    [341, 128]], dtype=int16)
                    ...],
                'dt_scores': [0.758478200014338, 0.7021546472698513, 0.8536622648391111, 0.8619181462164781, 0.8321051217096188, 0.8868756173427551, 0.7982964727675609, 0.8289939036796322, 0.8289428877522524, 0.8587063317632897, 0.7786755892491615, 0.8502032769081344, 0.8703346500042997, 0.834490931790065, 0.908291103353393, 0.7614978661708064, 0.8325774055997542, 0.7843421347676149, 0.8680889482955594, 0.8788859304537682, 0.8963341277518075, 0.9364654810069546, 0.8092413027028257, 0.8503743089091863, 0.7920740420391101, 0.7592224394793805, 0.7920547400069311, 0.6641757962457888, 0.8650289477605955, 0.8079483304467047, 0.8532207681055275, 0.8913377034754717],
                'rec_text': ['登机牌', 'BOARDING', 'PASS', '舱位', 'CLASS', '序号 SERIALNO.', '座位号', '日期 DATE', 'SEAT NO', '航班 FLIGHW', '035', 'MU2379', '始发地', 'FROM', '登机口', 'GATE', '登机时间BDT', '目的地TO', '福州', 'TAIYUAN', 'G11', 'FUZHOU', '身份识别IDNO', '姓名NAME', 'ZHANGQIWEI', 票号TKTNO', '张祺伟', '票价FARE', 'ETKT7813699238489/1', '登机口于起飞前10分钟关闭GATESCLOSE10MINUTESBEFOREDEPARTURETIME'],
                'rec_score': [0.9985831379890442, 0.999696917533874512, 0.9985735416412354, 0.9842517971992493, 0.9383274912834167, 0.9943678975105286, 0.9419361352920532, 0.9221674799919128, 0.9555020928382874, 0.9870321154594421, 0.9664073586463928, 0.9988052248954773, 0.9979352355003357, 0.9985110759735107, 0.9943482875823975, 0.9991195797920227, 0.9936401844024658, 0.9974591135978699, 0.9743705987930298, 0.9980487823486328, 0.9874696135520935, 0.9900962710380554, 0.9952947497367859, 0.9950481653213501, 0.989926815032959, 0.9915552139282227, 0.9938777685165405, 0.997239887714386, 0.9963340759277344, 0.9936134815216064, 0.97223961353302]}
                ```

            === "img"

                <p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/boardingpass.png"></p>

    === "Table Recognition"

        ```bash
        paddlex --pipeline table_recognition --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/table_recognition.jpg --device gpu:0
        ```

        ??? question "What's the result"
            === "output"
                ```bash
                {'input_path': '/root/.paddlex/predict_input/table_recognition.jpg', 'layout_result': {'input_path': '/root/.paddlex/predict_input/table_recognition.jpg', 'boxes': [{'cls_id': 3, 'label': 'Table', 'score': 0.6014542579650879, 'coordinate': [0, 21, 551, 118]}]}, 'ocr_result': {'dt_polys': [array([[37., 40.],
                    [75., 40.],
                    [75., 60.],
                    [37., 60.]], dtype=float32), array([[123.,  37.],
                    [200.,  37.],
                    [200.,  59.],
                    [123.,  59.]], dtype=float32), array([[227.,  37.],
                    [391.,  37.],
                    [391.,  58.],
                    [227.,  58.]], dtype=float32), array([[416.,  36.],
                    [535.,  38.],
                    [535.,  61.],
                    [415.,  58.]], dtype=float32), array([[35., 73.],
                    [78., 73.],
                    [78., 92.],
                    [35., 92.]], dtype=float32), array([[287.,  73.],
                    [328.,  73.],
                    [328.,  92.],
                    [287.,  92.]], dtype=float32), array([[453.,  72.],
                    [495.,  72.],
                    [495.,  94.],
                    [453.,  94.]], dtype=float32), array([[ 17., 103.],
                    [ 94., 103.],
                    [ 94., 118.],
                    [ 17., 118.]], dtype=float32), array([[145., 104.],
                    [178., 104.],
                    [178., 118.],
                    [145., 118.]], dtype=float32), array([[277., 104.],
                    [337., 102.],
                    [338., 118.],
                    [278., 118.]], dtype=float32), array([[446., 102.],
                    [504., 104.],
                    [503., 118.],
                    [445., 118.]], dtype=float32)], 'rec_text': ['Dres', '连续工作3', '取出来放在网上，没想', '江、整江等八大', 'Abstr', 'rSrivi', '$709.', 'cludingGiv', '2.72', 'Ingcubic', '$744.78'], 'rec_score': [0.9934158325195312, 0.9990204572677612, 0.9967061877250671, 0.9375461935997009, 0.9947397112846375, 0.9972746968269348, 0.9904290437698364, 0.973427414894104, 0.9983080625534058, 0.993423342704773, 0.9964120984077454], 'input_path': 'table_recognition.jpg'}, 'table_result': [{'input_path': 'table_recognition.jpg', 'layout_bbox': [0, 21, 551, 118], 'bbox': array([[  4.395736 ,  25.238262 , 113.31014  ,  25.316246 , 115.454315 ,
                        71.8867   ,   3.7177477,  71.7937   ],
                    [110.727455 ,  25.94007  , 210.07187  ,  26.028755 , 209.66394  ,
                        65.96484  , 109.59861  ,  66.09809  ],
                    [214.45381  ,  26.027939 , 407.95276  ,  26.112846 , 409.6684   ,
                        66.91336  , 215.27292  ,  67.002014 ],
                    [402.81863  ,  26.123789 , 549.03656  ,  26.231564 , 549.19995  ,
                        66.88339  , 404.48068  ,  66.74034  ],
                    [  2.4458022,  64.68588  , 102.7665   ,  65.10228  , 105.79447  ,
                        96.051254 ,   2.5367072,  95.35514  ],
                    [108.85877  ,  65.80549  , 211.70216  ,  66.02091  , 210.79245  ,
                        94.75581  , 107.59308  ,  94.42664  ],
                    [217.05621  ,  64.98496  , 407.76328  ,  65.133484 , 406.8436   ,
                        96.00133  , 214.67896  ,  95.87226  ],
                    [401.73572  ,  64.60494  , 547.9967   ,  64.73921  , 548.19135  ,
                        96.09901  , 402.26733  ,  95.95529  ],
                    [  2.4882016,  93.589554 , 107.01325  ,  93.67592  , 107.8446   ,
                        120.13259  ,   2.508764 , 119.85027  ],
                    [110.773125 ,  93.98633  , 213.354    ,  94.08046  , 212.46033  ,
                        120.80207  , 109.29008  , 120.613045 ],
                    [216.08781  ,  94.19984  , 405.843    ,  94.28341  , 405.9974   ,
                        121.33152  , 215.10301  , 121.299034 ],
                    [403.92212  ,  94.44883  , 548.30963  ,  94.54982  , 548.4949   ,
                        122.610176 , 404.53433  , 122.49881  ]], dtype=float32), 'img_idx': 0, 'html': '<html><body><table><tr><td>Dres</td><td>连续工作3</td><td>取出来放在网上，没想</td><td>江、整江等八大</td></tr><tr><td>Abstr</td><td></td><td>rSrivi</td><td>$709.</td></tr><tr><td>cludingGiv</td><td>2.72</td><td>Ingcubic</td><td>$744.78</td></tr></table></body></html>'}]}
                ```


            === "img"
                <p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/table_recognition/03.png"></p>

    === "Layout Parsing"

        ```bash
        paddlex --pipeline layout_parsing --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/demo_paper.png --device gpu:0
        ```

        ??? question "What's the result"
            ```bash
            {'input_path': PosixPath('/root/.paddlex/temp/tmp5jmloefs.png'), 'parsing_result': [{'input_path': PosixPath('/root/.paddlex/temp/tmpshsq8_w0.png'), 'layout_bbox': [51.46833, 74.22329, 542.4082, 232.77504], 'image': {'img': array([[[255, 255, 255],
                    [255, 255, 255],
                    [255, 255, 255],
                    ...,
                    [213, 221, 238],
                    [217, 223, 240],
                    [233, 234, 241]],

                [[255, 255, 255],
                    [255, 255, 255],
                    [255, 255, 255],
                    ...,
                    [255, 255, 255],
                    [255, 255, 255],
                    [255, 255, 255]]], dtype=uint8), 'image_text': ''}, 'layout': 'single'}, {'input_path': PosixPath('/root/.paddlex/temp/tmpcd2q9uyu.png'), 'layout_bbox': [47.68295, 243.08054, 546.28253, 295.71045], 'figure_title': 'Overview of RT-DETR, We feed th', 'layout': 'single'}, {'input_path': PosixPath('/root/.paddlex/temp/tmpr_iqa8b3.png'), 'layout_bbox': [58.416977, 304.1531, 275.9134, 400.07513], 'image': {'img': array([[[255, 255, 255],
                    [255, 255, 255],
                    [255, 255, 255],
                    ...,
                    [255, 255, 255],
                    [255, 255, 255],
                    [255, 255, 255]]], dtype=uint8), 'image_text': ''}, 'layout': 'left'}, {'input_path': PosixPath('/root/.paddlex/temp/tmphpblxl3p.png'), 'layout_bbox': [100.62961, 405.97458, 234.79774, 414.77414], 'figure_title': 'Figure 5. The fusion block in CCFF.', 'layout': 'left'}, {'input_path': PosixPath('/root/.paddlex/temp/tmplgnczrsf.png'), 'layout_bbox': [47.81724, 421.9041, 288.01566, 550.538], 'text': 'D, Ds, not only significantly reduces latency (35% faster),\nRut\nnproves accuracy (0.4% AP higher), CCFF is opti\nased on the cross-scale fusion module, which\nnsisting of convolutional lavers intc\npath.\nThe role of the fusion block is t\n into a new feature, and its\nFigure 5. The f\nblock contains tw\n1 x1\nchannels, /V RepBlock\n. anc\n: two-path outputs are fused by element-wise add. We\ntormulate the calculation ot the hvbrid encoder as:', 'layout': 'left'}, {'input_path': PosixPath('/root/.paddlex/temp/tmpsq0ey9md.png'), 'layout_bbox': [94.60716, 558.703, 288.04193, 600.19434], 'formula': '\\begin{array}{l}{{\\Theta=K=\\mathrm{p.s.sp{\\pm}}\\mathrm{i.s.s.}(\\mathrm{l.s.}(\\mathrm{l.s.}(\\mathrm{l.s.}}),{\\qquad\\mathrm{{a.s.}}\\mathrm{s.}}}\\\\ {{\\tau_{\\mathrm{{s.s.s.s.s.}}(\\mathrm{l.s.},\\mathrm{l.s.},\\mathrm{s.s.}}\\mathrm{s.}\\mathrm{s.}}\\end{array}),}}\\\\ {{\\bar{\\mathrm{e-c.c.s.s.}(\\mathrm{s.},\\mathrm{s.s.},\\ s_{s}}\\mathrm{s.s.},\\tau),}}\\end{array}', 'layout': 'left'}, {'input_path': PosixPath('/root/.paddlex/temp/tmpv30qy0v4.png'), 'layout_bbox': [47.975555, 607.12024, 288.5776, 629.1252], 'text': 'tened feature to the same shape as Ss.\nwhere Re shape represents restoring the shape of the flat-', 'layout': 'left'}, {'input_path': PosixPath('/root/.paddlex/temp/tmp0jejzwwv.png'), 'layout_bbox': [48.383354, 637.581, 245.96404, 648.20496], 'paragraph_title': '4.3. Uncertainty-minimal Query Selection', 'layout': 'left'}, {'input_path': PosixPath('/root/.paddlex/temp/tmpushex416.png'), 'layout_bbox': [47.80134, 656.002, 288.50192, 713.24994], 'text': 'To reduce the difficulty of optimizing object queries in\nDETR, several subsequent works [42, 44, 45] propose query\nselection schemes, which have in common that they use the\nconfidence score to select the top K’ features from the en-\ncoder to initialize object queries (or just position queries).', 'layout': 'left'}, {'input_path': PosixPath('/root/.paddlex/temp/tmpki7e_6wc.png'), 'layout_bbox': [306.6371, 302.1026, 546.3772, 419.76724], 'text': 'The confidence score represents the likelihood that the fea\nture includes foreground objects. Nevertheless, the \nare required to simultaneously model the category\nojects, both of which determine the quality of the\npertor\ncore of the fes\nBased on the analysis, the current query\n considerable level of uncertainty in the\nresulting in sub-optimal initialization for\nand hindering the performance of the detector.', 'layout': 'right'}, {'input_path': PosixPath('/root/.paddlex/temp/tmppbxrfehp.png'), 'layout_bbox': [306.0642, 422.7347, 546.9216, 539.45734], 'text': 'To address this problem, we propose the uncertainty mini\nmal query selection scheme, which explicitly const\noptim\n the epistemic uncertainty to model the\nfeatures, thereby providing \nhigh-quality\nr the decoder. Specifically,\n the discrepancy between i\nalization P\nand classificat\n.(2\ntunction for the gradie', 'layout': 'right'}, {'input_path': PosixPath('/root/.paddlex/temp/tmp1mgiyd21.png'), 'layout_bbox': [331.52808, 549.32635, 546.5229, 586.15546], 'formula': '\\begin{array}{c c c}{{}}&{{}}&{{\\begin{array}{c}{{i\\langle X\\rangle=({\\bar{Y}}({\\bar{X}})+{\\bar{Z}}({\\bar{X}})\\mid X\\in{\\bar{\\pi}}^{\\prime}}}&{{}}\\\\ {{}}&{{}}&{{}}\\end{array}}}&{{\\emptyset}}\\\\ {{}}&{{}}&{{C(\\bar{X},{\\bar{X}})=C..\\scriptstyle(\\bar{0},{\\bar{Y}})+{\\mathcal{L}}_{{\\mathrm{s}}}({\\bar{X}}),\\ 6)}}&{{}}\\end{array}', 'layout': 'right'}, {'input_path': PosixPath('/root/.paddlex/temp/tmp8t73dpym.png'), 'layout_bbox': [306.44016, 592.8762, 546.84314, 630.60126], 'text': 'where  and y denote the prediction and ground truth,\n= (c, b), c and b represent the category and bounding\nbox respectively, X represent the encoder feature.', 'layout': 'right'}, {'input_path': PosixPath('/root/.paddlex/temp/tmpftnxeyjm.png'), 'layout_bbox': [306.15652, 632.3142, 546.2463, 713.19073], 'text': 'Effectiveness analysis. To analyze the effectiveness of the\nuncertainty-minimal query selection, we visualize the clas-\nsificatior\nscores and IoU scores of the selected fe\nCOCO\na 12017, Figure 6. We draw the scatterplo\nt with\ndots\nrepresent the selected features from the model trained\nwith uncertainty-minimal query selection and vanilla query', 'layout': 'right'}]}
            ```

    === "Formula Recognition"

        ```bash
        paddlex --pipeline formula_recognition --input https://paddle-model-ecology.bj.bcebos.com/paddlex/demo_image/general_formula_recognition.png --device gpu:0
        ```

        ??? question "What's the result"
            === "output"
                ```bash
                {'input_path': '/root/.paddlex/predict_input/general_formula_recognition.png', 'layout_result': {'input_path': '/root/.paddlex/predict_input/general_formula_recognition.png', 'boxes': [{'cls_id': 3, 'label': 'number', 'score': 0.7580855488777161, 'coordinate': [1028.3635, 205.46213, 1038.953, 222.99033]}, {'cls_id': 0, 'label': 'paragraph_title', 'score': 0.8882032632827759, 'coordinate': [272.75305, 204.50894, 433.7473, 226.17996]}, {'cls_id': 2, 'label': 'text', 'score': 0.9685840606689453, 'coordinate': [272.75928, 282.17773, 1041.9316, 374.44687]}, {'cls_id': 2, 'label': 'text', 'score': 0.9559416770935059, 'coordinate': [272.39056, 385.54114, 1044.1521, 443.8598]}, {'cls_id': 2, 'label': 'text', 'score': 0.9610629081726074, 'coordinate': [272.40817, 467.2738, 1045.1033, 563.4855]}, {'cls_id': 7, 'label': 'formula', 'score': 0.8916195034980774, 'coordinate': [503.45743, 594.6236, 1040.6804, 619.73895]}, {'cls_id': 2, 'label': 'text', 'score': 0.973675549030304, 'coordinate': [272.32007, 648.8599, 1040.8702, 775.15686]}, {'cls_id': 7, 'label': 'formula', 'score': 0.9038916230201721, 'coordinate': [554.2307, 803.5825, 1040.4657, 855.3159]}, {'cls_id': 2, 'label': 'text', 'score': 0.9025381803512573, 'coordinate': [272.535, 875.1402, 573.1086, 898.3587]}, {'cls_id': 2, 'label': 'text', 'score': 0.8336610794067383, 'coordinate': [317.48013, 909.60864, 966.8498, 933.7868]}, {'cls_id': 2, 'label': 'text', 'score': 0.8779091238975525, 'coordinate': [19.704018, 653.322, 72.433235, 1215.1992]}, {'cls_id': 2, 'label': 'text', 'score': 0.8832409977912903, 'coordinate': [272.13028, 958.50806, 1039.7928, 1019.476]}, {'cls_id': 7, 'label': 'formula', 'score': 0.9088466167449951, 'coordinate': [517.1226, 1042.3978, 1040.2208, 1095.7457]}, {'cls_id': 2, 'label': 'text', 'score': 0.9587949514389038, 'coordinate': [272.03336, 1112.9269, 1041.0201, 1206.8417]}, {'cls_id': 2, 'label': 'text', 'score': 0.8885666131973267, 'coordinate': [271.7495, 1231.8752, 710.44495, 1255.7981]}, {'cls_id': 7, 'label': 'formula', 'score': 0.8907185196876526, 'coordinate': [581.2295, 1287.4525, 1039.8014, 1312.772]}, {'cls_id': 2, 'label': 'text', 'score': 0.9559596180915833, 'coordinate': [273.1827, 1341.421, 1041.0299, 1401.7255]}, {'cls_id': 2, 'label': 'text', 'score': 0.875311553478241, 'coordinate': [272.8338, 1427.3711, 789.7108, 1451.1359]}, {'cls_id': 7, 'label': 'formula', 'score': 0.9152213931083679, 'coordinate': [524.9582, 1474.8136, 1041.6333, 1530.7142]}, {'cls_id': 2, 'label': 'text', 'score': 0.9584835767745972, 'coordinate': [272.81665, 1549.524, 1042.9962, 1608.7157]}]}, 'ocr_result': {}, 'table_result': [], 'dt_polys': [array([[ 503.45743,  594.6236 ],
                    [1040.6804 ,  594.6236 ],
                    [1040.6804 ,  619.73895],
                    [ 503.45743,  619.73895]], dtype=float32), array([[ 554.2307,  803.5825],
                    [1040.4657,  803.5825],
                    [1040.4657,  855.3159],
                    [ 554.2307,  855.3159]], dtype=float32), array([[ 517.1226, 1042.3978],
                    [1040.2208, 1042.3978],
                    [1040.2208, 1095.7457],
                    [ 517.1226, 1095.7457]], dtype=float32), array([[ 581.2295, 1287.4525],
                    [1039.8014, 1287.4525],
                    [1039.8014, 1312.772 ],
                    [ 581.2295, 1312.772 ]], dtype=float32), array([[ 524.9582, 1474.8136],
                    [1041.6333, 1474.8136],
                    [1041.6333, 1530.7142],
                    [ 524.9582, 1530.7142]], dtype=float32)], 'rec_formula': ['F({\bf x})=C(F_{1}(x_{1}),\cdot\cdot\cdot,F_{N}(x_{N})).\qquad\qquad\qquad(1)', 'p(\mathbf{x})=c(\mathbf{u})\prod_{i}p(x_{i}).\qquad\qquad\qquad\qquad\qquad\quad\quad~~\quad~~~~~~~~~~~~~~~(2)', 'H_{c}({\bf x})=-\int_{{\bf{u}}}c({\bf{u}})\log c({\bf{u}})d{\bf{u}}.~~~~~~~~~~~~~~~~~~~~~(3)', 'I({\bf x})=-H_{c}({\bf x}).\qquad\qquad\qquad\qquad(4)', 'H({\bf x})=\sum_{i}H(x_{i})+H_{c}({\bf x}).\eqno\qquad\qquad\qquad(5)']}
                ```

            === "img"

                <p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/formula_recognition/02.jpg"></p>

    === "Seal Text Recognition"

        ```bash
        paddlex --pipeline seal_recognition --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/seal_text_det.png --device gpu:0
        ```

        ??? question "What's the result"
            === "output"
                ```bash
                    {'input_path': PosixPath('/root/.paddlex/temp/tmpa8eqnpus.png'), 'layout_result': {'input_path': PosixPath('/root/.paddlex/temp/tmpa8eqnpus.png'), 'boxes': [{'cls_id': 2, 'label': 'seal', 'score': 0.9813321828842163, 'coordinate': [0, 5.1820183, 639.59314, 637.7533]}]}, 'ocr_result': {'dt_polys': [array([[166, 468],
                        [206, 503],
                    [249, 523],
                    [312, 535],
                    [364, 529],
                    [390, 521],
                    [428, 505],
                    [465, 476],
                    [468, 474],
                    [473, 474],
                    [476, 475],
                    [478, 477],
                    [508, 507],
                    [510, 510],
                    [511, 514],
                    [509, 518],
                    [507, 521],
                    [458, 559],
                    [455, 560],
                    [399, 584],
                    [399, 584],
                    [369, 591],
                    [367, 592],
                    [308, 597],
                    [305, 596],
                    [240, 584],
                    [239, 584],
                    [220, 577],
                    [169, 552],
                    [166, 551],
                    [120, 510],
                    [117, 507],
                    [116, 503],
                    [117, 499],
                    [121, 495],
                    [153, 468],
                    [156, 467],
                    [161, 467]]), array([[439, 444],
                    [443, 444],
                    [446, 446],
                    [448, 448],
                    [450, 451],
                    [450, 454],
                    [448, 498],
                    [448, 502],
                    [445, 505],
                    [442, 507],
                    [439, 507],
                    [399, 505],
                    [196, 506],
                    [192, 505],
                    [189, 503],
                    [187, 500],
                    [187, 497],
                    [186, 458],
                    [186, 456],
                    [187, 451],
                    [188, 448],
                    [192, 444],
                    [194, 444],
                    [198, 443]]), array([[463, 347],
                    [468, 347],
                    [472, 350],
                    [474, 353],
                    [476, 360],
                    [477, 425],
                    [476, 429],
                    [474, 433],
                    [470, 436],
                    [466, 438],
                    [463, 438],
                    [175, 439],
                    [170, 438],
                    [166, 435],
                    [163, 432],
                    [161, 426],
                    [161, 361],
                    [161, 356],
                    [163, 352],
                    [167, 349],
                    [172, 347],
                    [184, 346],
                    [186, 346]]), array([[325,  38],
                    [485,  91],
                    [489,  94],
                    [493,  96],
                    [587, 225],
                    [588, 230],
                    [589, 234],
                    [592, 384],
                    [591, 389],
                    [588, 393],
                    [585, 397],
                    [581, 399],
                    [576, 399],
                    [572, 398],
                    [508, 380],
                    [503, 379],
                    [499, 375],
                    [498, 370],
                    [497, 367],
                    [493, 258],
                    [428, 171],
                    [421, 165],
                    [323, 136],
                    [225, 165],
                    [207, 175],
                    [144, 260],
                    [141, 365],
                    [141, 370],
                    [138, 374],
                    [134, 378],
                    [131, 379],
                    [ 66, 398],
                    [ 61, 398],
                    [ 56, 398],
                    [ 52, 395],
                    [ 48, 391],
                    [ 47, 386],
                    [ 47, 384],
                    [ 47, 235],
                    [ 48, 230],
                    [ 50, 226],
                    [146,  96],
                    [151,  92],
                    [154,  91],
                    [315,  38],
                    [320,  37]])], 'dt_scores': [0.99375725701319, 0.9871711582010613, 0.9937523531067023, 0.9911629231838204], 'rec_text': ['5263647368706', '吗繁物', '发票专天津君和缘商贸有限公司'], 'rec_score': [0.9933745265007019, 0.998288631439209, 0.9999362230300903, 0.9923253655433655], 'input_path': PosixPath('/Users/chenghong0temp/tmpa8eqnpus.png')}, 'src_file_name': 'https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/seal_text_det.png', 'page_id': 0}                
                    ```

            === "img"

                <p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/seal_recognition/03.png"></p>


!!! example "Computer VisionCLI"

    === "Image Classification"

        ```bash
        paddlex --pipeline image_classification --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg --device gpu:0
        ```

        ??? question "What's the result"
            === "output"
                ```bash
                {'input_path': '/root/.paddlex/predict_input/general_image_classification_001.jpg', 'class_ids': [296, 170, 356, 258, 248], 'scores': [0.62736, 0.03752, 0.03256, 0.0323, 0.03194], 'label_names': ['ice bear, polar bear, Ursus Maritimus, Thalarctos maritimus', 'Irish wolfhound', 'weasel', 'Samoyed, Samoyede', 'Eskimo dog, husky']}
                ```

            === "img"

                <p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/image_classification/03.png"></p>

    === "Object Detection"

        ```bash
        paddlex --pipeline object_detection --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_object_detection_002.png --device gpu:0
        ```
        ??? question "What's the result"
            === "output"
                ```bash
                {'input_path': '/root/.paddlex/predict_input/general_object_detection_002.png', 'boxes': [{'cls_id': 49, 'label': 'orange', 'score': 0.8188097476959229, 'coordinate': [661, 93, 870, 305]}, {'cls_id': 47, 'label': 'apple', 'score': 0.7743489146232605, 'coordinate': [76, 274, 330, 520]}, {'cls_id': 47, 'label': 'apple', 'score': 0.7270504236221313, 'coordinate': [285, 94, 469, 297]}, {'cls_id': 46, 'label': 'banana', 'score': 0.5570532083511353, 'coordinate': [310, 361, 685, 712]}, {'cls_id': 47, 'label': 'apple', 'score': 0.5484835505485535, 'coordinate': [764, 285, 924, 440]}, {'cls_id': 47, 'label': 'apple', 'score': 0.5160726308822632, 'coordinate': [853, 169, 987, 303]}, {'cls_id': 60, 'label': 'dining table', 'score': 0.5142655968666077, 'coordinate': [0, 0, 1072, 720]}, {'cls_id': 47, 'label': 'apple', 'score': 0.5101479291915894, 'coordinate': [57, 23, 213, 176]}]}
                ```

            === "img"

                <p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/object_detection/03.png"></p>

    === "Instance Segmentation"

        ```bash
        paddlex --pipeline instance_segmentation --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_instance_segmentation_004.png --device gpu:0
        ```

        ??? question "What's the result"
            === "output"
                ```bash
                {'input_path': '/root/.paddlex/predict_input/general_instance_segmentation_004.png', 'boxes': [{'cls_id': 0, 'label': 'person', 'score': 0.8698326945304871, 'coordinate': [339, 0, 639, 575]}, {'cls_id': 0, 'label': 'person', 'score': 0.8571141362190247, 'coordinate': [0, 0, 195, 575]}, {'cls_id': 0, 'label': 'person', 'score': 0.8202633857727051, 'coordinate': [88, 113, 401, 574]}, {'cls_id': 0, 'label': 'person', 'score': 0.7108577489852905, 'coordinate': [522, 21, 767, 574]}, {'cls_id': 27, 'label': 'tie', 'score': 0.554280698299408, 'coordinate': [247, 311, 355, 574]}]}
                ```

            === "img"

                <p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/instance_segmentation/03.png"></p>

    === "Semantic Segmentation"

        ```bash
        paddlex --pipeline semantic_segmentation --input https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/application/semantic_segmentation/makassaridn-road_demo.png --device gpu:0
        ```

        ??? question "What's the result"
            === "output"
                ```bash
                {'input_path': '/root/.paddlex/predict_input/makassaridn-road_demo.png', 'pred': '...'}
                ```

            === "img"

                <p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/semantic_segmentation/03.png"></p>

    === "Image Multi-label Classification"

        ```bash
        paddlex --pipeline multi_label_image_classification --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg --device gpu:0
        ```

        ??? question "What's the result"
            === "output"
                ```bash
                {'input_path': '/root/.paddlex/predict_input/general_image_classification_001.jpg', 'class_ids': [21, 0, 30, 24], 'scores': [0.99257, 0.70596, 0.63001, 0.57852], 'label_names': ['bear', 'person', 'skis', 'backpack']}
                ```

            === "img"

                <p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/image_multi_label_classification/02.png"></p>

    === "Small Object Detection"

        ```bash
        paddlex --pipeline small_object_detection --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/small_object_detection.jpg --device gpu:0
        ```

        ??? question "What's the result"
            === "output"
                ```bash
                {'input_path': '/root/.paddlex/predict_input/small_object_detection.jpg', 'boxes': [{'cls_id': 3, 'label': 'car', 'score': 0.9243856072425842, 'coordinate': [624, 638, 682, 741]}, {'cls_id': 3, 'label': 'car', 'score': 0.9206348061561584, 'coordinate': [242, 561, 356, 613]}, {'cls_id': 3, 'label': 'car', 'score': 0.9194547533988953, 'coordinate': [670, 367, 705, 400]}, {'cls_id': 3, 'label': 'car', 'score': 0.9162291288375854, 'coordinate': [459, 259, 523, 283]}, {'cls_id': 4, 'label': 'van', 'score': 0.9075379371643066, 'coordinate': [467, 213, 498, 242]}, {'cls_id': 4, 'label': 'van', 'score': 0.9066920876502991, 'coordinate': [547, 351, 577, 397]}, {'cls_id': 3, 'label': 'car', 'score': 0.9041045308113098, 'coordinate': [502, 632, 562, 736]}, {'cls_id': 3, 'label': 'car', 'score': 0.8934890627861023, 'coordinate': [613, 383, 647, 427]}, {'cls_id': 3, 'label': 'car', 'score': 0.8803309202194214, 'coordinate': [640, 280, 671, 309]}, {'cls_id': 3, 'label': 'car', 'score': 0.8727016448974609, 'coordinate': [1199, 256, 1259, 281]}, {'cls_id': 3, 'label': 'car', 'score': 0.8705748915672302, 'coordinate': [534, 410, 570, 461]}, {'cls_id': 3, 'label': 'car', 'score': 0.8654043078422546, 'coordinate': [669, 248, 702, 271]}, {'cls_id': 3, 'label': 'car', 'score': 0.8555219769477844, 'coordinate': [525, 243, 550, 270]}, {'cls_id': 3, 'label': 'car', 'score': 0.8522038459777832, 'coordinate': [526, 220, 553, 243]}, {'cls_id': 3, 'label': 'car', 'score': 0.8392605185508728, 'coordinate': [557, 141, 575, 158]}, {'cls_id': 3, 'label': 'car', 'score': 0.8353804349899292, 'coordinate': [537, 120, 553, 133]}, {'cls_id': 3, 'label': 'car', 'score': 0.8322211503982544, 'coordinate': [585, 132, 603, 147]}, {'cls_id': 3, 'label': 'car', 'score': 0.8298957943916321, 'coordinate': [701, 283, 736, 313]}, {'cls_id': 3, 'label': 'car', 'score': 0.8217393159866333, 'coordinate': [885, 347, 943, 377]}, {'cls_id': 3, 'label': 'car', 'score': 0.820313572883606, 'coordinate': [493, 150, 511, 168]}, {'cls_id': 0, 'label': 'pedestrian', 'score': 0.8183429837226868, 'coordinate': [203, 701, 224, 743]}, {'cls_id': 0, 'label': 'pedestrian', 'score': 0.815082848072052, 'coordinate': [185, 710, 201, 744]}, {'cls_id': 6, 'label': 'tricycle', 'score': 0.7892289757728577, 'coordinate': [311, 371, 344, 407]}, {'cls_id': 6, 'label': 'tricycle', 'score': 0.7812919020652771, 'coordinate': [345, 380, 388, 405]}, {'cls_id': 0, 'label': 'pedestrian', 'score': 0.7748346328735352, 'coordinate': [295, 500, 309, 532]}, {'cls_id': 0, 'label': 'pedestrian', 'score': 0.7688500285148621, 'coordinate': [851, 436, 863, 466]}, {'cls_id': 3, 'label': 'car', 'score': 0.7466475367546082, 'coordinate': [565, 114, 580, 128]}, {'cls_id': 3, 'label': 'car', 'score': 0.7156463265419006, 'coordinate': [483, 66, 495, 78]}, {'cls_id': 3, 'label': 'car', 'score': 0.704211950302124, 'coordinate': [607, 138, 642, 152]}, {'cls_id': 3, 'label': 'car', 'score': 0.7021926045417786, 'coordinate': [505, 72, 518, 83]}, {'cls_id': 0, 'label': 'pedestrian', 'score': 0.6897469162940979, 'coordinate': [802, 460, 815, 488]}, {'cls_id': 3, 'label': 'car', 'score': 0.671891450881958, 'coordinate': [574, 123, 593, 136]}, {'cls_id': 9, 'label': 'motorcycle', 'score': 0.6712754368782043, 'coordinate': [445, 317, 472, 334]}, {'cls_id': 0, 'label': 'pedestrian', 'score': 0.6695684790611267, 'coordinate': [479, 309, 489, 332]}, {'cls_id': 3, 'label': 'car', 'score': 0.6273623704910278, 'coordinate': [654, 210, 677, 234]}, {'cls_id': 3, 'label': 'car', 'score': 0.6070230603218079, 'coordinate': [640, 166, 667, 185]}, {'cls_id': 3, 'label': 'car', 'score': 0.6064521670341492, 'coordinate': [461, 59, 476, 71]}, {'cls_id': 3, 'label': 'car', 'score': 0.5860581398010254, 'coordinate': [464, 87, 484, 100]}, {'cls_id': 9, 'label': 'motorcycle', 'score': 0.5792551636695862, 'coordinate': [390, 390, 419, 408]}, {'cls_id': 3, 'label': 'car', 'score': 0.5559225678443909, 'coordinate': [481, 125, 496, 140]}, {'cls_id': 0, 'label': 'pedestrian', 'score': 0.5531904697418213, 'coordinate': [869, 306, 880, 331]}, {'cls_id': 0, 'label': 'pedestrian', 'score': 0.5468509793281555, 'coordinate': [895, 294, 904, 319]}, {'cls_id': 3, 'label': 'car', 'score': 0.5451828241348267, 'coordinate': [505, 94, 518, 108]}, {'cls_id': 3, 'label': 'car', 'score': 0.5398445725440979, 'coordinate': [657, 188, 681, 208]}, {'cls_id': 4, 'label': 'van', 'score': 0.5318890810012817, 'coordinate': [518, 88, 534, 102]}, {'cls_id': 3, 'label': 'car', 'score': 0.5296525359153748, 'coordinate': [527, 71, 540, 81]}, {'cls_id': 6, 'label': 'tricycle', 'score': 0.5168400406837463, 'coordinate': [528, 320, 563, 346]}, {'cls_id': 3, 'label': 'car', 'score': 0.5088561177253723, 'coordinate': [511, 84, 530, 95]}, {'cls_id': 0, 'label': 'pedestrian', 'score': 0.502006471157074, 'coordinate': [841, 266, 850, 283]}]}
                ```

            === "img"

                <p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/small_object_detection/02.png"></p>

    === "Image Anomaly Detection"

        ```bash
        paddlex --pipeline anomaly_detection --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/uad_grid.png --device gpu:0
        ```

        ??? question "What's the result"
            === "output"
                ```bash
                {'input_path': '/root/.paddlex/predict_input/uad_grid.png', 'pred': '...'}
                ```

            === "img"

                <p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/image_anomaly_detection/02.png"></p>

!!! example "Time Series-relatedCLI"

    === "Time Series Forecasting"

        ```bash
        paddlex --pipeline ts_fc --input https://paddle-model-ecology.bj.bcebos.com/paddlex/ts/demo_ts/ts_fc.csv --device gpu:0
        ```

        ??? question "What's the result"
            ```bash
            {'input_path': 'ts_fc.csv', 'forecast':                            OT
            date
            2018-06-26 20:00:00  9.586131
            2018-06-26 21:00:00  9.379762
            2018-06-26 22:00:00  9.252275
            2018-06-26 23:00:00  9.249993
            2018-06-27 00:00:00  9.164998
            ...                       ...
            2018-06-30 15:00:00  8.830340
            2018-06-30 16:00:00  9.291553
            2018-06-30 17:00:00  9.097666
            2018-06-30 18:00:00  8.905430
            2018-06-30 19:00:00  8.993793

            [96 rows x 1 columns]}
            ```

    === "Time Series Anomaly Detection"

        ```bash
        paddlex --pipeline ts_ad --input https://paddle-model-ecology.bj.bcebos.com/paddlex/ts/demo_ts/ts_ad.csv --device gpu:0
        ```

        ??? question "What's the result"
            ```bash
            {'input_path': 'ts_ad.csv', 'anomaly':            label
            timestamp
            220226         0
            220227         0
            220228         0
            220229         0
            220230         0
            ...          ...
            220317         1
            220318         1
            220319         1
            220320         1
            220321         0

            [96 rows x 1 columns]}
            ```

    === "Time Series Classification"

        ```bash
        paddlex --pipeline ts_cls --input https://paddle-model-ecology.bj.bcebos.com/paddlex/ts/demo_ts/ts_cls.csv --device gpu:0
        ```

        ??? question "What's the result"
            ```bash
            {'input_path': 'ts_cls.csv', 'classification':         classid     score
            sample
            0             0  0.617688}
            ```

## 📝 Python Usage

A few lines of code can complete the quick inference of the production line, with a unified Python script format as follows:

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline=[production line name])
output = pipeline.predict([input image name])
for res in output:
    res.print()
    res.save_to_img("./output/")
    res.save_to_json("./output/")
```
The following steps were executed:

* `create_pipeline()` instantiates the production line object
* Pass in the image and call the `predict` method of the production line object for inference prediction
* Process the prediction results


!!! example "OCR-related Python"

    === "OCR"

        ```python
        from paddlex import create_pipeline

        pipeline = create_pipeline(pipeline="OCR")

        output = pipeline.predict("general_ocr_002.png")
        for res in output:
            res.print()
            res.save_to_img("./output/")
        ```

    === "Table Recognition"

        ```python
        from paddlex import create_pipeline

        pipeline = create_pipeline(pipeline="table_recognition")

        output = pipeline.predict("table_recognition.jpg")
        for res in output:
            res.print()
            res.save_to_img("./output/")
            res.save_to_xlsx("./output/")
            res.save_to_html("./output/")
        ```

    === "Layout Parsing"

        ```python
        from paddlex import create_pipeline

        pipeline = create_pipeline(pipeline="layout_parsing")

        output = pipeline.predict("demo_paper.png")
        for res in output:
            res.print()
            res.save_to_img("./output/")
            res.save_to_xlsx("./output/")
            res.save_to_html("./output/")
        ```

    === "Formula Recognition"

        ```python
        from paddlex import create_pipeline

        pipeline = create_pipeline(pipeline="formula_recognition")

        output = pipeline.predict("general_formula_recognition.png")
        for res in output:
            res.print()
        ```

    === "Seal Text Recognition"

        ```python
        from paddlex import create_pipeline

        pipeline = create_pipeline(pipeline="seal_recognition")

        output = pipeline.predict("seal_text_det.png")
        for res in output:
            res.print()
            res.save_to_img("./output/")
        ```

!!! example "Computer Vision Python"

    === "Image Classification"

        ```python
        from paddlex import create_pipeline

        pipeline = create_pipeline(pipeline="image_classification")

        output = pipeline.predict("general_image_classification_001.jpg")
        for res in output:
            res.print()
            res.save_to_img("./output/")
            res.save_to_json("./output/")
        ```

    === "Object Detection"

        ```python
        from paddlex import create_pipeline

        pipeline = create_pipeline(pipeline="object_detection")

        output = pipeline.predict("general_object_detection_002.png")
        for res in output:
            res.print()
            res.save_to_img("./output/")
            res.save_to_json("./output/")
        ```

    === "Instance Segmentation"

        ```python
        from paddlex import create_pipeline

        pipeline = create_pipeline(pipeline="instance_segmentation")

        output = pipeline.predict("general_instance_segmentation_004.png")
        for res in output:
            res.print()
            res.save_to_img("./output/")
            res.save_to_json("./output/")
        ```

    === "Semantic Segmentation"

        ```python
        from paddlex import create_pipeline

        pipeline = create_pipeline(pipeline="semantic_segmentation")

        output = pipeline.predict("makassaridn-road_demo.png")
        for res in output:
            res.print()
            res.save_to_img("./output/")
            res.save_to_json("./output/")
        ```

    === "Image Multi-label Classification"

        ```python
        from paddlex import create_pipeline

        pipeline = create_pipeline(pipeline="multi_label_image_classification")

        output = pipeline.predict("general_image_classification_001.jpg")
        for res in output:
            res.print()
            res.save_to_img("./output/")
            res.save_to_json("./output/")
        ```

    === "Small Object Detection"

        ```python
        from paddlex import create_pipeline

        pipeline = create_pipeline(pipeline="small_object_detection")

        output = pipeline.predict("small_object_detection.jpg")
        for res in output:
            res.print()
            res.save_to_img("./output/")
            res.save_to_json("./output/")
        ```

    === "Image Anomaly Detection"

        ```python
        from paddlex import create_pipeline

        pipeline = create_pipeline(pipeline="anomaly_detection")

        output = pipeline.predict("uad_grid.png")
        for res in output:
            res.print()
            res.save_to_img("./output/")
            res.save_to_json("./output/")
        ```

!!! example "Time Series-related Python"

    === "Time Series Forecasting"

        ```python
        from paddlex import create_pipeline

        pipeline = create_pipeline(pipeline="ts_fc")

        output = pipeline.predict("ts_fc.csv")
        for res in output:
            res.print()
            res.save_to_csv("./output/")
        ```

    === "Time Series Anomaly Detection"

        ```python
        from paddlex import create_pipeline
        pipeline = create_pipeline(pipeline="./my_path/ts_ad.yaml")
        output = pipeline.predict("ts_ad.cs")
        for res in output:
            res.print()
            res.save_to_csv("./output/")
        ```

    === "Time Series Classification"

        ```python
        from paddlex import create_pipeline

        pipeline = create_pipeline(pipeline="ts_cls")

        output = pipeline.predict("ts_cls.csv")
        for res in output:
            res.print()
            res.save_to_csv("./output/")
        ```

## 🚀 Detailed Tutorials

<div class="grid cards" markdown>

- **Document Information Extraction**

    ---

    Document scene information extraction v3 (PP-ChatOCRv3) is a document and image intelligent analysis solution with PaddlePaddle features, combining LLM and OCR technologies to solve complex document information extraction challenges such as layout analysis, rare character recognition, multi-page PDF, table, and seal recognition in one stop.

    [:octicons-arrow-right-24: Tutorial](pipeline_usage/tutorials/information_extraction_pipelines/document_scene_information_extraction.en.md)

- **OCR**

    ---

    The general OCR production line is used to solve text recognition tasks, extract text information from images, and output it in text form. Based on the end-to-end OCR system, it can achieve millisecond-level precise text content prediction on CPUs and reach open-source SOTA in general scenarios.

    [:octicons-arrow-right-24: Tutorial](pipeline_usage/tutorials/ocr_pipelines/OCR.en.md)

- **Image Classification**

    ---

    Image classification can automatically extract image features and classify them accurately, recognizing various objects such as animals, plants, traffic signs, etc., and is widely used in object recognition, scene understanding, and automatic tagging fields.

    [:octicons-arrow-right-24: Tutorial](pipeline_usage/tutorials/cv_pipelines/image_classification.en.md)

- **Object Detection**

    ---

    Object detection aims to identify the categories and locations of multiple objects in images or videos by generating bounding boxes to mark these objects. This technology is widely used in fields such as autonomous driving, surveillance systems, and smart photo albums.

    [:octicons-arrow-right-24: Tutorial](pipeline_usage/tutorials/cv_pipelines/object_detection.en.md)

- **Small Object Detection**

    ---

    Small object detection is a technology specifically designed to recognize smaller objects in images, widely used in surveillance, unmanned driving, and satellite image analysis fields. It can accurately locate and classify small-sized objects such as pedestrians, traffic signs, or small animals from complex scenes.

    [:octicons-arrow-right-24: Tutorial](pipeline_usage/tutorials/cv_pipelines/small_object_detection.en.md)

- **Time Series Forecasting**

    ---

    Time series forecasting is a technique that uses historical data to predict future trends by analyzing the patterns of change in time series data. It is widely used in financial markets, weather forecasting, and sales forecasting fields.

    [:octicons-arrow-right-24: Tutorial](pipeline_usage/tutorials/time_series_pipelines/time_series_forecasting.en.md)

</div>

[:octicons-arrow-right-24: More](pipeline_usage/pipeline_develop_guide.en.md)


## 💬 Discussion

We warmly welcome and encourage community members to raise questions, share ideas, and feedback in the [Discussions](https://github.com/PaddlePaddle/PaddleX/discussions) section. Whether you want to report a bug, discuss a feature request, seek help, or just want to keep up with the latest project news, this is a great platform.
