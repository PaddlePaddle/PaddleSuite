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

import ultra_infer as ui
import numpy as np
from paddlex.inference.results import InstanceSegResult
from paddlex.modules.instance_segmentation.model_list import MODELS
from pydantic import BaseModel

from paddlex_hpi._utils.typing import BatchData, Data
from paddlex_hpi.models.base import CVPredictor, HPIParams


class _InstanceSegPPParams(BaseModel):
    threshold: float
    label_list: List[str]


class InstanceSegPredictor(CVPredictor):
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

    def _build_ui_model(
        self, option: ui.RuntimeOption
    ) -> ui.vision.detection.PaddleDetectionModel:
        model = ui.vision.detection.PaddleDetectionModel(
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
            instance_seg_result = self._create_instance_seg_result(data, ui_result)
            results.append({"result": instance_seg_result})
        return results

    def _get_pp_params(self) -> _InstanceSegPPParams:
        return _InstanceSegPPParams(
            threshold=self.config["draw_threshold"],
            label_list=self.config["label_list"],
        )

    def _create_instance_seg_result(
        self, data: Data, ui_result: Any
    ) -> InstanceSegResult:
        inds = sorted(
            range(len(ui_result.scores)), key=ui_result.scores.__getitem__, reverse=True
        )
        inds = [i for i in inds if ui_result.scores[i] > self._pp_params.threshold]
        inds = [i for i in inds if ui_result.label_ids[i] > -1]
        ids = [ui_result.label_ids[i] for i in inds]
        scores = [ui_result.scores[i] for i in inds]
        boxes = [ui_result.boxes[i] for i in inds]
        masks = [ui_result.masks[i] for i in inds]
        masks = [
            np.array(mask.data, dtype=np.uint8).reshape(mask.shape) for mask in masks
        ]
        dic = {
            "input_path": data["input_path"],
            "boxes": [
                {
                    "cls_id": id_,
                    "label": self._pp_params.label_list[id_],
                    "score": score,
                    "coordinate": box,
                }
                for id_, score, box in zip(ids, scores, boxes)
            ],
            "masks": masks,
        }
        return InstanceSegResult(dic)
