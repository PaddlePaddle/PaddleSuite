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

import os
from typing import Any, Dict, List, Optional, Union

import ultrainfer as ui
import numpy as np
from paddlex.inference.results import MLClassResult
from paddlex.modules.multilabel_classification.model_list import MODELS

from paddlex_hpi._utils.typing import BatchData, Data
from paddlex_hpi.models.base import CVPredictor, HPIParams


class MLClasPredictor(CVPredictor):
    entities = MODELS

    def __init__(
        self,
        model_dir: Union[str, os.PathLike],
        config: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
        hpi_params: Optional[HPIParams] = None,
    ) -> None:
        super().__init__(
            model_dir=model_dir,
            config=config,
            device=device,
            hpi_params=hpi_params,
        )
        self._label_list = self._get_label_list()

    def _build_ui_model(
        self, option: ui.RuntimeOption
    ) -> ui.vision.classification.PyOnlyMultilabelClassificationModel:
        model = ui.vision.classification.PyOnlyMultilabelClassificationModel(
            str(self.model_path),
            str(self.params_path),
            str(self.config_path),
            runtime_option=option,
        )
        return model

    def _predict(self, batch_data: BatchData) -> BatchData:
        imgs = [np.ascontiguousarray(data["img"]) for data in batch_data]
        ui_results = self._ui_model.batch_predict(imgs)
        results: BatchData = []
        for data, ui_result in zip(batch_data, ui_results):
            ml_clas_result = self._create_ml_clas_result(data, ui_result)
            results.append({"result": ml_clas_result})
        return results

    def _get_label_list(self) -> Optional[List[str]]:
        pp_config = self.config["PostProcess"]
        if "MultiLabelThreshOutput" not in pp_config:
            raise RuntimeError("`MultiLabelThreshOutput` config not found")
        label_list = pp_config["MultiLabelThreshOutput"].get("label_list", None)
        return label_list

    def _create_ml_clas_result(self, data: Data, ui_result: Any) -> MLClassResult:
        dic = {
            "input_path": data["input_path"],
            "class_ids": ui_result.label_ids,
            "scores": np.around(ui_result.scores, decimals=5).tolist(),
        }
        if self._label_list is not None:
            dic["label_names"] = [self._label_list[i] for i in ui_result.label_ids]
        return MLClassResult(dic)
