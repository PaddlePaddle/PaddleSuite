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

#include <vector>

#include "ultra_infer/ultra_infer_model.h"
#include "ultra_infer/vision/common/processors/transform.h"
#include "ultra_infer/vision/common/result.h"

#include "ultra_infer/utils/unique_ptr.h"
#include "ultra_infer/vision/ocr/ppocr/dbdetector.h"
#include "ultra_infer/vision/ocr/ppocr/recognizer.h"
#include "ultra_infer/vision/ocr/ppocr/structurev2_table.h"
#include "ultra_infer/vision/ocr/ppocr/utils/ocr_postprocess_op.h"

namespace ultra_infer {
/** \brief This pipeline can launch detection model, classification model and
 * recognition model sequentially. All OCR pipeline APIs are defined inside this
 * namespace.
 *
 */
namespace pipeline {
/*! @brief PPStructureV2Table is used to load PP-OCRv2 series models provided by
 * PaddleOCR.
 */
class ULTRAINFER_DECL PPStructureV2Table : public UltraInferModel {
public:
  /** \brief Set up the detection model path, recognition model path and table
   * model path respectively.
   *
   * \param[in] det_model Path of detection model, e.g ./ch_PP-OCRv2_det_infer
   * \param[in] rec_model Path of recognition model, e.g ./ch_PP-OCRv2_rec_infer
   * \param[in] table_model Path of table recognition model, e.g
   * ./en_ppstructure_mobile_v2.0_SLANet_infer
   */
  PPStructureV2Table(ultra_infer::vision::ocr::DBDetector *det_model,
                     ultra_infer::vision::ocr::Recognizer *rec_model,
                     ultra_infer::vision::ocr::StructureV2Table *table_model);

  /** \brief Clone a new PPStructureV2Table with less memory usage when multiple
   * instances of the same model are created
   *
   * \return new PPStructureV2Table* type unique pointer
   */
  std::unique_ptr<PPStructureV2Table> Clone() const;

  /** \brief Predict the input image and get OCR result.
   *
   * \param[in] im The input image data, comes from cv::imread(), is a 3-D array
   * with layout HWC, BGR format. \param[in] result The output OCR result will
   * be writen to this structure. \return true if the prediction successed,
   * otherwise false.
   */
  virtual bool Predict(cv::Mat *img, ultra_infer::vision::OCRResult *result);
  virtual bool Predict(const cv::Mat &img,
                       ultra_infer::vision::OCRResult *result);
  /** \brief BatchPredict the input image and get OCR result.
   *
   * \param[in] images The list of input image data, comes from cv::imread(), is
   * a 3-D array with layout HWC, BGR format. \param[in] batch_result The output
   * list of OCR result will be writen to this structure. \return true if the
   * prediction successed, otherwise false.
   */
  virtual bool
  BatchPredict(const std::vector<cv::Mat> &images,
               std::vector<ultra_infer::vision::OCRResult> *batch_result);

  bool Initialized() const override;
  bool SetRecBatchSize(int rec_batch_size);
  int GetRecBatchSize();

protected:
  ultra_infer::vision::ocr::DBDetector *detector_ = nullptr;
  ultra_infer::vision::ocr::Recognizer *recognizer_ = nullptr;
  ultra_infer::vision::ocr::StructureV2Table *table_ = nullptr;

private:
  int rec_batch_size_ = 6;
};

namespace application {
namespace ocrsystem {
typedef pipeline::PPStructureV2Table PPStructureV2TableSystem;
} // namespace ocrsystem
} // namespace application

} // namespace pipeline
} // namespace ultra_infer
