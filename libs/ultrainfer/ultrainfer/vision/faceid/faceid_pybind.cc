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

namespace ultrainfer {
void BindInsightFace(pybind11::module &m);
void BindAdaFace(pybind11::module &m);
void BindFaceId(pybind11::module &m) {
  auto faceid_module = m.def_submodule("faceid", "Face recognition models.");
  BindInsightFace(faceid_module);
  BindAdaFace(faceid_module);
}
} // namespace ultrainfer
