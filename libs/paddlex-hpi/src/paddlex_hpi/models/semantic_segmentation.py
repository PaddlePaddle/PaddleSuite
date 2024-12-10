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

from typing import Any, List

import xdeploy as xd
import numpy as np
from paddlex.inference.results import SegResult
from paddlex.modules.semantic_segmentation.model_list import MODELS

from paddlex_hpi._utils.typing import BatchData, Data
from paddlex_hpi.models.base import CVPredictor


class SegPredictor(CVPredictor):
    entities = MODELS

    def _build_xd_model(
        self, option: xd.RuntimeOption
    ) -> xd.vision.segmentation.PaddleSegModel:
        model = xd.vision.segmentation.PaddleSegModel(
            str(self.model_path),
            str(self.params_path),
            str(self.config_path),
            runtime_option=option,
        )
        return model

    def _predict(self, batch_data: BatchData) -> BatchData:
        imgs = [np.ascontiguousarray(data["img"]) for data in batch_data]
        xd_results = self._xd_model.batch_predict(imgs)
        results: BatchData = []
        for data, xd_result in zip(batch_data, xd_results):
            seg_result = self._create_seg_result(data, xd_result)
            results.append({"result": seg_result})
        return results

    def _create_seg_result(self, data: Data, xd_result: Any) -> SegResult:
        pred = np.array(xd_result.label_map, dtype=np.int32).reshape(xd_result.shape)
        pred = pred[np.newaxis]
        dic = {
            "input_path": data["input_path"],
            "pred": pred,
        }
        return SegResult(dic)
