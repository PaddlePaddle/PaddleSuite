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
from paddlex.inference.common.batch_sampler import ImageBatchSampler
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

    def _build_batch_sampler(self) -> ImageBatchSampler:
        """Builds and returns an ImageBatchSampler instance.

        Returns:
            ImageBatchSampler: An instance of ImageBatchSampler.
        """
        return ImageBatchSampler()

    def _get_result_class(self) -> type:
        """Returns the result class, TopkResult.

        Returns:
            type: The TopkResult class.
        """
        return TopkResult
    
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

    def process(self, batch_data: List[Any]) -> Dict[str, List[Any]]:
        batch_raw_imgs = self._data_reader(imgs=batch_data)
        imgs = [np.ascontiguousarray(img) for img in batch_raw_imgs]
        ui_results = self._ui_model.batch_predict(imgs)

        class_ids = []
        scores = []
        label_names = []
        for ui_result in ui_results:
            class_ids.append(ui_result.label_ids)
            scores.append(np.around(ui_result.scores, decimals=5).tolist())
            if self._pp_params.label_list is not None:
                label_names.append([self._pp_params.label_list[i] for i in ui_result.label_ids])

        return {
            "input_path": batch_data,
            "input_img": batch_raw_imgs,
            "class_ids": class_ids,
            "scores": scores,
            "label_names": label_names,
        }

    def _get_pp_params(self) -> _ClasPPParams:
        pp_config = self.config["PostProcess"]
        if "Topk" not in pp_config:
            raise RuntimeError("`Topk` config not found")
        topk_config = pp_config["Topk"]
        topk = topk_config["topk"]
        label_list = topk_config.get("label_list", None)
        return _ClasPPParams(topk=topk, label_list=label_list)
