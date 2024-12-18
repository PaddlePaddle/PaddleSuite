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

import tempfile
from typing import Any, List

import ultrainfer as ui
import numpy as np
from paddlex.inference.results import TableRecResult
from paddlex.modules.table_recognition.model_list import MODELS

from paddlex_hpi._utils.compat import get_compat_version
from paddlex_hpi._utils.typing import BatchData, Data
from paddlex_hpi.models.base import CVPredictor


class TablePredictor(CVPredictor):
    entities = MODELS

    def _build_ui_model(
        self, option: ui.RuntimeOption
    ) -> ui.vision.ocr.StructureV2Table:
        compat_version = get_compat_version()
        if compat_version == "2.5" or self.model_name == "SLANet":
            bbox_shape_type = "ori"
        else:
            bbox_shape_type = "pad"
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt") as f:
            pp_config = self.config["PostProcess"]
            for lab in pp_config["character_dict"]:
                f.write(lab + "\n")
            f.flush()
            model = ui.vision.ocr.StructureV2Table(
                str(self.model_path),
                str(self.params_path),
                table_char_dict_path=f.name,
                box_shape=bbox_shape_type,
                runtime_option=option,
            )
        return model

    def _predict(self, batch_data: BatchData) -> BatchData:
        imgs = [np.ascontiguousarray(data["img"]) for data in batch_data]
        ui_results = self._ui_model.batch_predict(imgs)
        results: BatchData = []
        for data, ui_result in zip(batch_data, ui_results):
            table_result = self._create_table_result(data, ui_result)
            results.append({"result": table_result})
        return results

    def _create_table_result(self, data: Data, ui_result: Any) -> TableRecResult:
        dic = {
            "input_path": data["input_path"],
            "bbox": ui_result.table_boxes,
            "structure": ui_result.table_structure,
        }
        return TableRecResult(dic)
