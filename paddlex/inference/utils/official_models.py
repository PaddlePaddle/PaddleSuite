# copyright (c) 2024 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

from ...utils import logging
from ...utils.cache import CACHE_DIR
from ...utils.download import download_and_extract


OFFICIAL_MODELS = {
    "ResNet18": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ResNet18_infer.tar",
    "ResNet18_vd": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ResNet18_vd_infer.tar",
    "ResNet34": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ResNet34_infer.tar",
    "ResNet34_vd": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ResNet34_vd_infer.tar",
    "ResNet50": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ResNet50_infer.tar",
    "ResNet50_vd": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ResNet50_vd_infer.tar",
    "ResNet101": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ResNet101_infer.tar",
    "ResNet101_vd": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ResNet101_vd_infer.tar",
    "ResNet152": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ResNet152_infer.tar",
    "ResNet152_vd": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ResNet152_vd_infer.tar",
    "ResNet200_vd": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ResNet200_vd_infer.tar",
    "PP-LCNet_x0_25": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNet_x0_25_infer.tar",
    "PP-LCNet_x0_25_textline_ori": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNet_x0_25_textline_ori_infer.tar",
    "PP-LCNet_x0_35": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNet_x0_35_infer.tar",
    "PP-LCNet_x0_5": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNet_x0_5_infer.tar",
    "PP-LCNet_x0_75": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNet_x0_75_infer.tar",
    "PP-LCNet_x1_0": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNet_x1_0_infer.tar",
    "PP-LCNet_x1_0_doc_ori": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNet_x1_0_doc_ori_infer.tar",
    "PP-LCNet_x1_5": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNet_x1_5_infer.tar",
    "PP-LCNet_x2_5": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNet_x2_5_infer.tar",
    "PP-LCNet_x2_0": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNet_x2_0_infer.tar",
    "PP-LCNetV2_small": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNetV2_small_infer.tar",
    "PP-LCNetV2_base": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNetV2_base_infer.tar",
    "PP-LCNetV2_large": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNetV2_large_infer.tar",
    "MobileNetV3_large_x0_35": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/\
MobileNetV3_large_x0_35_infer.tar",
    "MobileNetV3_large_x0_5": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/\
MobileNetV3_large_x0_5_infer.tar",
    "MobileNetV3_large_x0_75": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/\
MobileNetV3_large_x0_75_infer.tar",
    "MobileNetV3_large_x1_0": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/\
MobileNetV3_large_x1_0_infer.tar",
    "MobileNetV3_large_x1_25": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/\
MobileNetV3_large_x1_25_infer.tar",
    "MobileNetV3_small_x0_35": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/\
MobileNetV3_small_x0_35_infer.tar",
    "MobileNetV3_small_x0_5": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/\
MobileNetV3_small_x0_5_infer.tar",
    "MobileNetV3_small_x0_75": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/\
MobileNetV3_small_x0_75_infer.tar",
    "MobileNetV3_small_x1_0": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/\
MobileNetV3_small_x1_0_infer.tar",
    "MobileNetV3_small_x1_25": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/\
MobileNetV3_small_x1_25_infer.tar",
    "ConvNeXt_tiny": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ConvNeXt_tiny_infer.tar",
    "ConvNeXt_small": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ConvNeXt_small_infer.tar",
    "ConvNeXt_base_224": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ConvNeXt_base_224_infer.tar",
    "ConvNeXt_base_384": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ConvNeXt_base_384_infer.tar",
    "ConvNeXt_large_224": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ConvNeXt_large_224_infer.tar",
    "ConvNeXt_large_384": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ConvNeXt_large_384_infer.tar",
    "MobileNetV2_x0_25": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/\
MobileNetV2_x0_25_infer.tar",
    "MobileNetV2_x0_5": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV2_x0_5_infer.tar",
    "MobileNetV2_x1_0": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV2_x1_0_infer.tar",
    "MobileNetV2_x1_5": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV2_x1_5_infer.tar",
    "MobileNetV2_x2_0": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV2_x2_0_infer.tar",
    "MobileNetV1_x0_25": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/\
MobileNetV1_x0_25_infer.tar",
    "MobileNetV1_x0_5": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/\
MobileNetV1_x0_5_infer.tar",
    "MobileNetV1_x0_75": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/\
MobileNetV1_x0_75_infer.tar",
    "MobileNetV1_x1_0": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/\
MobileNetV1_x1_0_infer.tar",
    "SwinTransformer_tiny_patch4_window7_224": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/\
SwinTransformer_tiny_patch4_window7_224_infer.tar",
    "SwinTransformer_small_patch4_window7_224": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/\
SwinTransformer_small_patch4_window7_224_infer.tar",
    "SwinTransformer_base_patch4_window7_224": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/\
SwinTransformer_base_patch4_window7_224_infer.tar",
    "SwinTransformer_base_patch4_window12_384": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/\
SwinTransformer_base_patch4_window12_384_infer.tar",
    "SwinTransformer_large_patch4_window7_224": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/\
SwinTransformer_large_patch4_window7_224_infer.tar",
    "SwinTransformer_large_patch4_window12_384": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/\
SwinTransformer_large_patch4_window12_384_infer.tar",
    "PP-HGNet_tiny": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNet_tiny_infer.tar",
    "PP-HGNet_small": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNet_small_infer.tar",
    "PP-HGNet_base": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNet_base_infer.tar",
    "PP-HGNetV2-B0": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNetV2-B0_infer.tar",
    "PP-HGNetV2-B1": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNetV2-B1_infer.tar",
    "PP-HGNetV2-B2": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNetV2-B2_infer.tar",
    "PP-HGNetV2-B3": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNetV2-B3_infer.tar",
    "PP-HGNetV2-B4": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNetV2-B4_infer.tar",
    "PP-HGNetV2-B5": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNetV2-B5_infer.tar",
    "PP-HGNetV2-B6": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNetV2-B6_infer.tar",
    "FasterNet-L": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterNet-L_infer.tar",
    "FasterNet-M": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterNet-M_infer.tar",
    "FasterNet-S": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterNet-S_infer.tar",
    "FasterNet-T0": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterNet-T0_infer.tar",
    "FasterNet-T1": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterNet-T1_infer.tar",
    "FasterNet-T2": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterNet-T2_infer.tar",
    "StarNet-S1": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/StarNet-S1_infer.tar",
    "StarNet-S2": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/StarNet-S2_infer.tar",
    "StarNet-S3": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/StarNet-S3_infer.tar",
    "StarNet-S4": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/StarNet-S4_infer.tar",
    "MobileNetV4_conv_small": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV4_conv_small_infer.tar",
    "MobileNetV4_conv_medium": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV4_conv_medium_infer.tar",
    "MobileNetV4_conv_large": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV4_conv_large_infer.tar",
    "MobileNetV4_hybrid_medium": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV4_hybrid_medium_infer.tar",
    "MobileNetV4_hybrid_large": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV4_hybrid_large_infer.tar",
    "CLIP_vit_base_patch16_224": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/\
CLIP_vit_base_patch16_224_infer.tar",
    "CLIP_vit_large_patch14_224": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/\
CLIP_vit_large_patch14_224_infer.tar",
    "PP-LCNet_x1_0_ML": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNet_x1_0_ML_infer.tar",
    "PP-HGNetV2-B0_ML": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNetV2-B0_ML_infer.tar",
    "PP-HGNetV2-B4_ML": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNetV2-B4_ML_infer.tar",
    "PP-HGNetV2-B6_ML": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNetV2-B6_ML_infer.tar",
    "ResNet50_ML": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ResNet50_ML_infer.tar",
    "CLIP_vit_base_patch16_448_ML": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/CLIP_vit_base_patch16_448_ML_infer.tar",
    "PP-YOLOE_plus-X": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE_plus-X_infer.tar",
    "PP-YOLOE_plus-L": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE_plus-L_infer.tar",
    "PP-YOLOE_plus-M": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE_plus-M_infer.tar",
    "PP-YOLOE_plus-S": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE_plus-S_infer.tar",
    "RT-DETR-L": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/RT-DETR-L_infer.tar",
    "RT-DETR-H": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/RT-DETR-H_infer.tar",
    "RT-DETR-X": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/RT-DETR-X_infer.tar",
    "YOLOv3-DarkNet53": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/YOLOv3-DarkNet53_infer.tar",
    "YOLOv3-MobileNetV3": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/YOLOv3-MobileNetV3_infer.tar",
    "YOLOv3-ResNet50_vd_DCN": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/YOLOv3-ResNet50_vd_DCN_infer.tar",
    "YOLOX-L": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/YOLOX-L_infer.tar",
    "YOLOX-M": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/YOLOX-M_infer.tar",
    "YOLOX-N": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/YOLOX-N_infer.tar",
    "YOLOX-S": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/YOLOX-S_infer.tar",
    "YOLOX-T": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/YOLOX-T_infer.tar",
    "YOLOX-X": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/YOLOX-X_infer.tar",
    "RT-DETR-R18": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/RT-DETR-R18_infer.tar",
    "RT-DETR-R50": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/RT-DETR-R50_infer.tar",
    "PicoDet-S": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet-S_infer.tar",
    "PicoDet-L": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet-L_infer.tar",
    "Deeplabv3-R50": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Deeplabv3-R50_infer.tar",
    "Deeplabv3-R101": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Deeplabv3-R101_infer.tar",
    "Deeplabv3_Plus-R50": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/\
Deeplabv3_Plus-R50_infer.tar",
    "Deeplabv3_Plus-R101": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/\
Deeplabv3_Plus-R101_infer.tar",
    "PP-ShiTuV2_rec": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-ShiTuV2_rec_infer.tar",
    "PP-ShiTuV2_rec_CLIP_vit_base": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/\
PP-ShiTuV2_rec_CLIP_vit_base_infer.tar",
    "PP-ShiTuV2_rec_CLIP_vit_large": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/\
PP-ShiTuV2_rec_CLIP_vit_large_infer.tar",
    "PP-LiteSeg-T": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LiteSeg-T_infer.tar",
    "PP-LiteSeg-B": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LiteSeg-B_infer.tar",
    "OCRNet_HRNet-W48": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/OCRNet_HRNet-W48_infer.tar",
    "OCRNet_HRNet-W18": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/OCRNet_HRNet-W18_infer.tar",
    "SegFormer-B0": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SegFormer-B0_infer.tar",
    "SegFormer-B1": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SegFormer-B1_infer.tar",
    "SegFormer-B2": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SegFormer-B2_infer.tar",
    "SegFormer-B3": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SegFormer-B3_infer.tar",
    "SegFormer-B4": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SegFormer-B4_infer.tar",
    "SegFormer-B5": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SegFormer-B5_infer.tar",
    "SeaFormer_tiny": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SeaFormer_tiny_infer.tar",
    "SeaFormer_small": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SeaFormer_small_infer.tar",
    "SeaFormer_base": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SeaFormer_base_infer.tar",
    "SeaFormer_large": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SeaFormer_large_infer.tar",
    "Mask-RT-DETR-H": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Mask-RT-DETR-H_infer.tar",
    "Mask-RT-DETR-L": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Mask-RT-DETR-L_infer.tar",
    "PP-OCRv4_server_rec": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/\
PP-OCRv4_server_rec_infer.tar",
    "Mask-RT-DETR-S": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Mask-RT-DETR-S_infer.tar",
    "Mask-RT-DETR-M": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Mask-RT-DETR-M_infer.tar",
    "Mask-RT-DETR-X": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Mask-RT-DETR-X_infer.tar",
    "SOLOv2": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SOLOv2_infer.tar",
    "MaskRCNN-ResNet50": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MaskRCNN-ResNet50_infer.tar",
    "MaskRCNN-ResNet50-FPN": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MaskRCNN-ResNet50-FPN_infer.tar",
    "MaskRCNN-ResNet50-vd-FPN": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MaskRCNN-ResNet50-vd-FPN_infer.tar",
    "MaskRCNN-ResNet101-FPN": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MaskRCNN-ResNet101-FPN_infer.tar",
    "MaskRCNN-ResNet101-vd-FPN": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MaskRCNN-ResNet101-vd-FPN_infer.tar",
    "MaskRCNN-ResNeXt101-vd-FPN": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MaskRCNN-ResNeXt101-vd-FPN_infer.tar",
    "Cascade-MaskRCNN-ResNet50-FPN": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Cascade-MaskRCNN-ResNet50-FPN_infer.tar",
    "Cascade-MaskRCNN-ResNet50-vd-SSLDv2-FPN": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Cascade-MaskRCNN-ResNet50-vd-SSLDv2-FPN_infer.tar",
    "PP-YOLOE_seg-S": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE_seg-S_infer.tar",
    "PP-OCRv3_mobile_rec": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/\
PP-OCRv3_mobile_rec_infer.tar",
    "en_PP-OCRv3_mobile_rec": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/\
en_PP-OCRv3_mobile_rec_infer.tar",
    "korean_PP-OCRv3_mobile_rec": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/\
korean_PP-OCRv3_mobile_rec_infer.tar",
    "japan_PP-OCRv3_mobile_rec": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/\
japan_PP-OCRv3_mobile_rec_infer.tar",
    "chinese_cht_PP-OCRv3_mobile_rec": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/\
chinese_cht_PP-OCRv3_mobile_rec_infer.tar",
    "te_PP-OCRv3_mobile_rec": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/\
te_PP-OCRv3_mobile_rec_infer.tar",
    "ka_PP-OCRv3_mobile_rec": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/\
ka_PP-OCRv3_mobile_rec_infer.tar",
    "ta_PP-OCRv3_mobile_rec": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/\
ta_PP-OCRv3_mobile_rec_infer.tar",
    "latin_PP-OCRv3_mobile_rec": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/\
latin_PP-OCRv3_mobile_rec_infer.tar",
    "arabic_PP-OCRv3_mobile_rec": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/\
arabic_PP-OCRv3_mobile_rec_infer.tar",
    "cyrillic_PP-OCRv3_mobile_rec": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/\
cyrillic_PP-OCRv3_mobile_rec_infer.tar",
    "devanagari_PP-OCRv3_mobile_rec": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/\
devanagari_PP-OCRv3_mobile_rec_infer.tar",
    "en_PP-OCRv4_mobile_rec": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/\
en_PP-OCRv4_mobile_rec_infer.tar",
    "PP-OCRv4_server_rec_doc": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/\
PP-OCRv4_server_rec_doc_infer.tar",
    "PP-OCRv4_mobile_rec": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/\
PP-OCRv4_mobile_rec_infer.tar",
    "PP-OCRv4_server_det": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/\
PP-OCRv4_server_det_infer.tar",
    "PP-OCRv4_mobile_det": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/\
PP-OCRv4_mobile_det_infer.tar",
    "PP-OCRv3_server_det": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/\
PP-OCRv3_server_det_infer.tar",
    "PP-OCRv3_mobile_det": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/\
PP-OCRv3_mobile_det_infer.tar",
    "PP-OCRv4_server_seal_det": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/\
PP-OCRv4_server_seal_det_infer.tar",
    "PP-OCRv4_mobile_seal_det": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/\
PP-OCRv4_mobile_seal_det_infer.tar",
    "ch_RepSVTR_rec": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/\
ch_RepSVTR_rec_infer.tar",
    "ch_SVTRv2_rec": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/\
ch_SVTRv2_rec_infer.tar",
    "PP-LCNet_x1_0_pedestrian_attribute": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/\
PP-LCNet_x1_0_pedestrian_attribute_infer.tar",
    "PP-LCNet_x1_0_vehicle_attribute": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/\
PP-LCNet_x1_0_vehicle_attribute_infer.tar",
    "PicoDet_layout_1x": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet_layout_1x_infer.tar",
    "PicoDet_layout_1x_table": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet_layout_1x_table_infer.tar",
    "SLANet": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SLANet_infer.tar",
    "SLANet_plus": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SLANet_plus_infer.tar",
    "LaTeX_OCR_rec": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/LaTeX_OCR_rec_infer.tar",
    "UniMERNet": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/UniMERNet_infer.tar",
    "PP-FormulaNet-S": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-FormulaNet-S_infer.tar",
    "PP-FormulaNet-L": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-FormulaNet-L_infer.tar",
    "FasterRCNN-ResNet34-FPN": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterRCNN-ResNet34-FPN_infer.tar",
    "FasterRCNN-ResNet50": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterRCNN-ResNet50_infer.tar",
    "FasterRCNN-ResNet50-FPN": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterRCNN-ResNet50-FPN_infer.tar",
    "FasterRCNN-ResNet50-vd-FPN": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterRCNN-ResNet50-vd-FPN_infer.tar",
    "FasterRCNN-ResNet50-vd-SSLDv2-FPN": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterRCNN-ResNet50-vd-SSLDv2-FPN_infer.tar",
    "FasterRCNN-ResNet101": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterRCNN-ResNet101_infer.tar",
    "FasterRCNN-ResNet101-FPN": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterRCNN-ResNet101-FPN_infer.tar",
    "FasterRCNN-ResNeXt101-vd-FPN": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterRCNN-ResNeXt101-vd-FPN_infer.tar",
    "FasterRCNN-Swin-Tiny-FPN": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterRCNN-Swin-Tiny-FPN_infer.tar",
    "Cascade-FasterRCNN-ResNet50-FPN": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Cascade-FasterRCNN-ResNet50-FPN_infer.tar",
    "Cascade-FasterRCNN-ResNet50-vd-SSLDv2-FPN": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Cascade-FasterRCNN-ResNet50-vd-SSLDv2-FPN_infer.tar",
    "UVDoc": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/UVDoc_infer.tar",
    "DLinear": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/DLinear_infer.tar",
    "NLinear": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/NLinear_infer.tar",
    "RLinear": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/RLinear_infer.tar",
    "Nonstationary": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Nonstationary_infer.tar",
    "TimesNet": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/TimesNet_infer.tar",
    "TiDE": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/TiDE_infer.tar",
    "PatchTST": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PatchTST_infer.tar",
    "DLinear_ad": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/DLinear_ad_infer.tar",
    "AutoEncoder_ad": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/AutoEncoder_ad_infer.tar",
    "Nonstationary_ad": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Nonstationary_ad_infer.tar",
    "PatchTST_ad": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PatchTST_ad_infer.tar",
    "TimesNet_ad": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/TimesNet_ad_infer.tar",
    "TimesNet_cls": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/TimesNet_cls_infer.tar",
    "STFPM": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/STFPM_infer.tar",
    "FCOS-ResNet50": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FCOS-ResNet50_infer.tar",
    "DETR-R50": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/DETR-R50_infer.tar",
    "PP-YOLOE-L_vehicle": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE-L_vehicle_infer.tar",
    "PP-YOLOE-S_vehicle": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE-S_vehicle_infer.tar",
    "PP-ShiTuV2_det": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-ShiTuV2_det_infer.tar",
    "PP-YOLOE-S_human": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE-S_human_infer.tar",
    "PP-YOLOE-L_human": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE-L_human_infer.tar",
    "PicoDet-M": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet-M_infer.tar",
    "PicoDet-XS": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet-XS_infer.tar",
    "PP-YOLOE_plus_SOD-L": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE_plus_SOD-L_infer.tar",
    "PP-YOLOE_plus_SOD-S": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE_plus_SOD-S_infer.tar",
    "PP-YOLOE_plus_SOD-largesize-L": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE_plus_SOD-largesize-L_infer.tar",
    "CenterNet-DLA-34": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/CenterNet-DLA-34_infer.tar",
    "CenterNet-ResNet50": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/CenterNet-ResNet50_infer.tar",
    "PicoDet-S_layout_3cls": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet-S_layout_3cls_infer.tar",
    "PicoDet-S_layout_17cls": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet-S_layout_17cls_infer.tar",
    "PicoDet-L_layout_3cls": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet-L_layout_3cls_infer.tar",
    "PicoDet-L_layout_17cls": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet-L_layout_17cls_infer.tar",
    "RT-DETR-H_layout_3cls": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/RT-DETR-H_layout_3cls_infer.tar",
    "RT-DETR-H_layout_17cls": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/RT-DETR-H_layout_17cls_infer.tar",
    "PicoDet_LCNet_x2_5_face": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet_LCNet_x2_5_face_infer.tar",
    "BlazeFace": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/BlazeFace_infer.tar",
    "BlazeFace-FPN-SSH": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/BlazeFace-FPN-SSH_infer.tar",
    "PP-YOLOE_plus-S_face": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE_plus-S_face_infer.tar",
    "MobileFaceNet": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileFaceNet_infer.tar",
    "ResNet50_face": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ResNet50_face_infer.tar",
    "PP-YOLOE-R_L": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE-R_L_infer.tar",
    "Co-Deformable-DETR-R50": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Co-Deformable-DETR-R50_infer.tar",
    "Co-Deformable-DETR-Swin-T": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Co-Deformable-DETR-Swin-T_infer.tar",
    "Co-DINO-R50": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Co-DINO-R50_infer.tar",
    "Co-DINO-Swin-L": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Co-DINO-Swin-L_infer.tar",
    "whisper_large": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/whisper_large.tar",
    "whisper_base": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/whisper_base.tar",
    "whisper_medium": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/whisper_medium.tar",
    "whisper_small": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/whisper_small.tar",
    "whisper_tiny": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/whisper_tiny.tar",
    "PP-TSM-R50_8frames_uniform": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-TSM-R50_8frames_uniform_infer.tar",
    "PP-TSMv2-LCNetV2_8frames_uniform": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-TSMv2-LCNetV2_8frames_uniform_infer.tar",
    "PP-TSMv2-LCNetV2_16frames_uniform": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-TSMv2-LCNetV2_16frames_uniform_infer.tar",
    "MaskFormer_tiny": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MaskFormer_tiny_infer.tar",
    "MaskFormer_small": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MaskFormer_small_infer.tar",
    "PP-LCNet_x1_0_table_cls": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNet_x1_0_table_cls_infer.tar",
    "SLANeXt_wired": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SLANeXt_wired_infer.tar",
    "SLANeXt_wireless": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SLANeXt_wireless_infer.tar",
    "RT-DETR-L_wired_table_cell_det": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/RT-DETR-L_wired_table_cell_det_infer.tar",
    "RT-DETR-L_wireless_table_cell_det": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/RT-DETR-L_wireless_table_cell_det_infer.tar",
    "YOWO": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/YOWO_infer.tar",
    "PP-TinyPose_128x96": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-TinyPose_128x96_infer.tar",
    "PP-TinyPose_256x192": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-TinyPose_256x192_infer.tar",
    "GroundingDINO-T": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/GroundingDINO-T_infer.tar",
    "SAM-H_box": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SAM-H_box_infer.tar",
    "SAM-H_point": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SAM-H_point_infer.tar",
}


class OfficialModelsDict(dict):
    """Official Models Dict"""

    def __getitem__(self, key):
        url = super().__getitem__(key)
        save_dir = Path(CACHE_DIR) / "official_models"
        logging.info(
            f"Using official model ({key}), the model files will be be automatically downloaded and saved in {save_dir}."
        )
        download_and_extract(url, save_dir, f"{key}", overwrite=False)
        return save_dir / f"{key}"


official_models = OfficialModelsDict(OFFICIAL_MODELS)
