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

import ultra_infer as ui
import numpy as np
from paddlex.inference.results import BaseResult
from paddlex.modules.general_recognition.model_list import MODELS

from paddlex_hpi._utils.typing import BatchData, Data
from paddlex_hpi.models.base import CVPredictor


class ShiTuRecPredictor(CVPredictor):
    entities = MODELS

    def _build_ui_model(
        self, option: ui.RuntimeOption
    ) -> ui.vision.classification.PPShiTuV2Recognizer:
        model = ui.vision.classification.PPShiTuV2Recognizer(
            str(self.model_path),
            str(self.params_path),
            str(self.config_path),
            runtime_option=option,
        )
        return model

    def _predict(self, batch_data: BatchData) -> BatchData:
        imgs = [
            np.ascontiguousarray(data["img"]).astype("float32") for data in batch_data
        ]
        ui_results = self._ui_model.batch_predict(imgs)
        results: BatchData = []
        for data, ui_result in zip(batch_data, ui_results):
            clas_result = self._create_rec_result(data, ui_result)
            results.append({"result": clas_result})
        return results

    def _create_rec_result(self, data: Data, ui_result: Any) -> BaseResult:
        dic = {
            "input_path": data["input_path"],
            "feature": np.array(ui_result.feature, dtype="float32"),
        }
        return BaseResult(dic)
