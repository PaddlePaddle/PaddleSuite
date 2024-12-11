// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once

#include "ultrainfer/core/config.h"
#ifdef ENABLE_VISION
#include "ultrainfer/vision/classification/contrib/resnet.h"
#include "ultrainfer/vision/classification/contrib/yolov5cls/yolov5cls.h"
#include "ultrainfer/vision/classification/ppcls/model.h"
#include "ultrainfer/vision/classification/ppshitu/ppshituv2_det.h"
#include "ultrainfer/vision/classification/ppshitu/ppshituv2_rec.h"
#include "ultrainfer/vision/detection/contrib/fastestdet/fastestdet.h"
#include "ultrainfer/vision/detection/contrib/nanodet_plus.h"
#include "ultrainfer/vision/detection/contrib/rknpu2/model.h"
#include "ultrainfer/vision/detection/contrib/scaledyolov4.h"
#include "ultrainfer/vision/detection/contrib/yolor.h"
#include "ultrainfer/vision/detection/contrib/yolov5/yolov5.h"
#include "ultrainfer/vision/detection/contrib/yolov5lite.h"
#include "ultrainfer/vision/detection/contrib/yolov5seg/yolov5seg.h"
#include "ultrainfer/vision/detection/contrib/yolov6.h"
#include "ultrainfer/vision/detection/contrib/yolov7/yolov7.h"
#include "ultrainfer/vision/detection/contrib/yolov7end2end_ort.h"
#include "ultrainfer/vision/detection/contrib/yolov7end2end_trt.h"
#include "ultrainfer/vision/detection/contrib/yolov8/yolov8.h"
#include "ultrainfer/vision/detection/contrib/yolox.h"
#include "ultrainfer/vision/detection/ppdet/model.h"
#include "ultrainfer/vision/facealign/contrib/face_landmark_1000.h"
#include "ultrainfer/vision/facealign/contrib/pfld.h"
#include "ultrainfer/vision/facealign/contrib/pipnet.h"
#include "ultrainfer/vision/facedet/contrib/centerface/centerface.h"
#include "ultrainfer/vision/facedet/contrib/retinaface.h"
#include "ultrainfer/vision/facedet/contrib/scrfd.h"
#include "ultrainfer/vision/facedet/contrib/ultraface.h"
#include "ultrainfer/vision/facedet/contrib/yolov5face.h"
#include "ultrainfer/vision/facedet/contrib/yolov7face/yolov7face.h"
#include "ultrainfer/vision/facedet/ppdet/blazeface/blazeface.h"
#include "ultrainfer/vision/faceid/contrib/adaface/adaface.h"
#include "ultrainfer/vision/faceid/contrib/insightface/model.h"
#include "ultrainfer/vision/generation/contrib/animegan.h"
#include "ultrainfer/vision/headpose/contrib/fsanet.h"
#include "ultrainfer/vision/keypointdet/pptinypose/pptinypose.h"
#include "ultrainfer/vision/matting/contrib/modnet.h"
#include "ultrainfer/vision/matting/contrib/rvm.h"
#include "ultrainfer/vision/matting/ppmatting/ppmatting.h"
#include "ultrainfer/vision/ocr/ppocr/classifier.h"
#include "ultrainfer/vision/ocr/ppocr/dbcurvedetector.h"
#include "ultrainfer/vision/ocr/ppocr/dbdetector.h"
#include "ultrainfer/vision/ocr/ppocr/ppocr_v2.h"
#include "ultrainfer/vision/ocr/ppocr/ppocr_v3.h"
#include "ultrainfer/vision/ocr/ppocr/ppocr_v4.h"
#include "ultrainfer/vision/ocr/ppocr/ppstructurev2_layout.h"
#include "ultrainfer/vision/ocr/ppocr/ppstructurev2_table.h"
#include "ultrainfer/vision/ocr/ppocr/recognizer.h"
#include "ultrainfer/vision/ocr/ppocr/structurev2_layout.h"
#include "ultrainfer/vision/ocr/ppocr/structurev2_ser_vi_layoutxlm.h"
#include "ultrainfer/vision/ocr/ppocr/structurev2_table.h"
#include "ultrainfer/vision/ocr/ppocr/utils/ocr_utils.h"
#include "ultrainfer/vision/ocr/ppocr/uvdocwarpper.h"
#include "ultrainfer/vision/perception/paddle3d/caddn/caddn.h"
#include "ultrainfer/vision/perception/paddle3d/centerpoint/centerpoint.h"
#include "ultrainfer/vision/perception/paddle3d/petr/petr.h"
#include "ultrainfer/vision/perception/paddle3d/smoke/smoke.h"
#include "ultrainfer/vision/segmentation/ppseg/model.h"
#include "ultrainfer/vision/sr/ppsr/model.h"
#include "ultrainfer/vision/tracking/pptracking/model.h"

#endif

#include "ultrainfer/vision/visualize/visualize.h"
