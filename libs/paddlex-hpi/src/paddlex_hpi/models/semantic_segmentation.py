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

from typing import Any, Dict, List

import ultra_infer as ui
import numpy as np
from paddlex.inference.common.batch_sampler import ImageBatchSampler
from paddlex.inference.results import SegResult
from paddlex.modules.semantic_segmentation.model_list import MODELS

from paddlex_hpi.models.base import CVPredictor


class SegPredictor(CVPredictor):
    entities = MODELS

    def _build_ui_model(
        self, option: ui.RuntimeOption
    ) -> ui.vision.segmentation.PaddleSegModel:
        model = ui.vision.segmentation.PaddleSegModel(
            str(self.model_path),
            str(self.params_path),
            str(self.config_path),
            runtime_option=option,
        )
        return model

    def _build_batch_sampler(self) -> ImageBatchSampler:
        return ImageBatchSampler()

    def _get_result_class(self) -> type:
        return SegResult

    def process(self, batch_data: List[Any]) -> Dict[str, List[Any]]:
        batch_raw_imgs = self._data_reader(imgs=batch_data)
        imgs = [np.ascontiguousarray(img) for img in batch_raw_imgs]
        ui_results = self._ui_model.batch_predict(imgs)

        batch_preds = []
        for ui_result in ui_results:
            pred = np.array(ui_result.label_map, dtype=np.int32).reshape(
                ui_result.shape
            )
            pred = pred[np.newaxis]
            batch_preds.append(pred)

        return {
            "input_path": batch_data,
            "input_img": batch_raw_imgs,
            "pred": batch_preds,
        }
