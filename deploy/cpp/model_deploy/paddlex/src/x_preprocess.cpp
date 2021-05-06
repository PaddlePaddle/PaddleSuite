// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "model_deploy/paddlex/include/x_preprocess.h"

namespace PaddleDeploy {

bool XPreprocess::Init(const YAML::Node& yaml_config) {
  if (!BuildTransform(yaml_config)) {
    return false;
  }
  model_type_ = yaml_config["model_type"].as<std::string>();
  model_name_ = yaml_config["model_name"].as<std::string>();
  return true;
}

bool XPreprocess::PrepareInputs(const std::vector<ShapeInfo>& shape_infos,
                                std::vector<cv::Mat>* imgs,
                                std::vector<DataBlob>* inputs,
                                int thread_num) {
  inputs->clear();
  if (!PreprocessImages(shape_infos, imgs, thread_num = thread_num)) {
    std::cerr << "Error happend while execute function "
              << "XPreprocess::Run" << std::endl;
    return false;
  }

  if (model_type_ == "detector") {
    return PrepareInputsForDetector(*imgs, shape_infos, inputs, thread_num);
  } else if (model_type_ == "segmenter") {
    return PrepareInputsForSegmenter(*imgs, shape_infos, inputs, thread_num);
  } else if (model_type_ == "classifier") {
    return PrepareInputsForClassifier(*imgs, shape_infos, inputs, thread_num);
  } else {
    std::cerr << "[ERROR] Unexpected model_type: "
              << model_type_ << " for PaddleXModel"
              << std::endl;
    return false;
  }
  return false;
}

bool XPreprocess::PrepareInputsForDetector(
                        const std::vector<cv::Mat>& imgs,
                        const std::vector<ShapeInfo>& shape_infos,
                        std::vector<DataBlob>* inputs,
                        int thread_num) {
  if (model_name_ == "FasterRCNN" || model_name_ == "MaskRCNN") {
    return PrepareInputsForRCNN(imgs, shape_infos, inputs, thread_num);
  } else if (model_name_ == "YOLOv3" || model_name_ == "PPYOLO") {
    return PrepareInputsForYOLO(imgs, shape_infos, inputs, thread_num);
  } else {
    std::cerr << "[ERROR] Unexpected model_name: '"
              << model_name_ << "' for PaddleX"
              << std::endl;
    return false;
  }
  return true;
}

bool XPreprocess::PrepareInputsForSegmenter(
                        const std::vector<cv::Mat>& imgs,
                        const std::vector<ShapeInfo>& shape_infos,
                        std::vector<DataBlob>* inputs,
                        int thread_num) {
  DataBlob im("image");
  int batch = imgs.size();
  int w = shape_infos[0].shapes.back()[0];
  int h = shape_infos[0].shapes.back()[1];
  im.Resize({batch, 3, h, w}, FLOAT32);
  int sample_shape = 3 * h * w;

  #pragma omp parallel for num_threads(thread_num)
  for (auto i = 0; i < batch; ++i) {
    memcpy(im.data.data() + i * sample_shape * sizeof(float),
            imgs[i].data, sample_shape * sizeof(float));
  }
  inputs->clear();
  inputs->push_back(std::move(im));
  return true;
}

bool XPreprocess::PrepareInputsForClassifier(
                        const std::vector<cv::Mat>& imgs,
                        const std::vector<ShapeInfo>& shape_infos,
                        std::vector<DataBlob>* inputs,
                        int thread_num) {
  DataBlob im("image");
  int batch = imgs.size();
  int w = shape_infos[0].shapes.back()[0];
  int h = shape_infos[0].shapes.back()[1];
  im.Resize({batch, 3, h, w}, FLOAT32);
  int sample_shape = 3 * h * w;

  #pragma omp parallel for num_threads(thread_num)
  for (auto i = 0; i < batch; ++i) {
    memcpy(im.data.data() + i * sample_shape * sizeof(float),
            imgs[i].data, sample_shape * sizeof(float));
  }
  inputs->clear();
  inputs->push_back(std::move(im));
  return true;
}

bool XPreprocess::PrepareInputsForYOLO(
    const std::vector<cv::Mat>& imgs, const std::vector<ShapeInfo>& shape_infos,
    std::vector<DataBlob>* inputs, int thread_num) {
  DataBlob im("image");
  DataBlob im_size("im_size");
  // TODO(jiangjiajun): only 3 channel supported
  int batch = imgs.size();
  int w = shape_infos[0].shapes.back()[0];
  int h = shape_infos[0].shapes.back()[1];

  im.Resize({batch, 3, h, w}, FLOAT32);
  im_size.Resize({batch, 2}, INT32);

  int sample_shape = 3 * h * w;
  #pragma omp parallel for num_threads(thread_num)
  for (auto i = 0; i < batch; ++i) {
    memcpy(im.data.data() + i * sample_shape * sizeof(float), imgs[i].data,
           sample_shape * sizeof(float));
    int data[2] = {shape_infos[i].shapes[0][1], shape_infos[i].shapes[0][0]};
    memcpy(im_size.data.data() + i * 2 * sizeof(int), data, 2 * sizeof(int));
  }

  inputs->clear();
  inputs->push_back(std::move(im));
  inputs->push_back(std::move(im_size));
  return true;
}

bool XPreprocess::PrepareInputsForRCNN(
    const std::vector<cv::Mat>& imgs, const std::vector<ShapeInfo>& shape_infos,
    std::vector<DataBlob>* inputs, int thread_num) {
  DataBlob im("image");
  DataBlob im_info("im_info");
  DataBlob im_shape("im_shape");
  // TODO(jiangjiajun): only 3 channel supported
  int batch = imgs.size();
  int w = shape_infos[0].shapes.back()[0];
  int h = shape_infos[0].shapes.back()[1];

  im.Resize({batch, 3, h, w}, FLOAT32);
  im_info.Resize({batch, 3}, FLOAT32);
  im_shape.Resize({batch, 3}, FLOAT32);

  int sample_shape = 3 * h * w;
  #pragma omp parallel for num_threads(thread_num)
  for (auto i = 0; i < batch; ++i) {
    int shapes_num = shape_infos[i].shapes.size();
    float origin_w = static_cast<float>(shape_infos[i].shapes[0][0]);
    float origin_h = static_cast<float>(shape_infos[i].shapes[0][1]);
    float resize_w = origin_w;
    for (auto j = shapes_num - 1; j > 1; --j) {
      if (shape_infos[i].transforms[j] == "Padding") {
        continue;
      }
      resize_w = static_cast<float>(shape_infos[i].shapes[j][0]);
      break;
    }
    float scale = resize_w / origin_w;
    float im_info_data[] = {static_cast<float>(h), static_cast<float>(w),
                            scale};
    float im_shape_data[] = {origin_h, origin_w, 1.0};
    memcpy(im.data.data() + i * sample_shape * sizeof(float), imgs[i].data,
           sample_shape * sizeof(float));
    memcpy(im_info.data.data() + i * 3 * sizeof(float), im_info_data,
           3 * sizeof(float));
    memcpy(im_shape.data.data() + i * 3 * sizeof(float), im_shape_data,
           3 * sizeof(float));
  }

  inputs->clear();
  inputs->push_back(std::move(im));
  inputs->push_back(std::move(im_info));
  inputs->push_back(std::move(im_shape));
  return true;
}

bool XPreprocess::Run(std::vector<cv::Mat>* imgs,
                        std::vector<DataBlob>* inputs,
                        std::vector<ShapeInfo>* shape_infos, int thread_num) {
  if (!ShapeInfer(*imgs, shape_infos, thread_num)) {
    std::cerr << "ShapeInfer failed while call XPreprocess::Run" << std::endl;
    return false;
  }
  if (!PrepareInputs(*shape_infos, imgs, inputs, thread_num)) {
    std::cerr << "PrepareInputs failed while call "
              << "XPreprocess::PrepareInputs" << std::endl;
    return false;
  }
  return true;
}

}  //  namespace PaddleDeploy
