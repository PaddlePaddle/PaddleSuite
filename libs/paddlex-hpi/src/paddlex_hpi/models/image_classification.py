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
from paddlex.inference.results import TopkResult
from paddlex.modules.image_classification.model_list import MODELS
from pydantic import BaseModel

from paddlex_hpi._utils.typing import BatchData, Data
from paddlex_hpi.models.base import CVPredictor, HPIParams


class _ClasPPParams(BaseModel):
    topk: int
    label_list: Optional[List[str]] = None


class ClasPredictor(CVPredictor):
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
        self._pp_params = self._get_pp_params()
        self._ui_model.postprocessor.topk = self._pp_params.topk

    def _build_ui_model(
        self, option: ui.RuntimeOption
    ) -> ui.vision.classification.PaddleClasModel:
        model = ui.vision.classification.PaddleClasModel(
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
            clas_result = self._create_clas_result(data, ui_result)
            results.append({"result": clas_result})
        return results

    def _get_pp_params(self) -> _ClasPPParams:
        pp_config = self.config["PostProcess"]
        if "Topk" not in pp_config:
            raise RuntimeError("`Topk` config not found")
        topk_config = pp_config["Topk"]
        topk = topk_config["topk"]
        label_list = topk_config.get("label_list", None)
        return _ClasPPParams(topk=topk, label_list=label_list)

    def _create_clas_result(self, data: Data, ui_result: Any) -> TopkResult:
        dic = {
            "input_path": data["input_path"],
            "class_ids": ui_result.label_ids,
            "scores": np.around(ui_result.scores, decimals=5).tolist(),
        }
        if self._pp_params.label_list is not None:
            dic["label_names"] = [
                self._pp_params.label_list[i] for i in ui_result.label_ids
            ]
        return TopkResult(dic)
