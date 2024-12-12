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
#include "ultrainfer/pybind/main.h"
#include "ultrainfer/runtime/backends/rknpu2/option.h"
namespace ultrainfer {
void BindRKNPU2Option(pybind11::module &m) {
  pybind11::enum_<ultrainfer::rknpu2::CpuName>(
      m, "CpuName", pybind11::arithmetic(), "CpuName for inference.")
      .value("RK356X", ultrainfer::rknpu2::CpuName::RK356X)
      .value("RK3588", ultrainfer::rknpu2::CpuName::RK3588)
      .value("UNDEFINED", ultrainfer::rknpu2::CpuName::UNDEFINED);
  pybind11::enum_<ultrainfer::rknpu2::CoreMask>(
      m, "CoreMask", pybind11::arithmetic(), "CoreMask for inference.")
      .value("RKNN_NPU_CORE_AUTO",
             ultrainfer::rknpu2::CoreMask::RKNN_NPU_CORE_AUTO)
      .value("RKNN_NPU_CORE_0", ultrainfer::rknpu2::CoreMask::RKNN_NPU_CORE_0)
      .value("RKNN_NPU_CORE_1", ultrainfer::rknpu2::CoreMask::RKNN_NPU_CORE_1)
      .value("RKNN_NPU_CORE_2", ultrainfer::rknpu2::CoreMask::RKNN_NPU_CORE_2)
      .value("RKNN_NPU_CORE_0_1",
             ultrainfer::rknpu2::CoreMask::RKNN_NPU_CORE_0_1)
      .value("RKNN_NPU_CORE_0_1_2",
             ultrainfer::rknpu2::CoreMask::RKNN_NPU_CORE_0_1_2)
      .value("RKNN_NPU_CORE_UNDEFINED",
             ultrainfer::rknpu2::CoreMask::RKNN_NPU_CORE_UNDEFINED);
}
} // namespace ultrainfer
