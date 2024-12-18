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
from paddlex.inference.results import TextDetResult
from paddlex.modules.text_detection.model_list import CURVE_MODELS, MODELS

from paddlex_hpi._utils.misc import parse_scale
from paddlex_hpi._utils.typing import BatchData, Data
from paddlex_hpi.models.base import CVPredictor, HPIParams


class TextDetPredictor(CVPredictor):
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

    # HACK
    @property
    def _is_curve_model(self) -> bool:
        return self.model_name in CURVE_MODELS

    def _build_ui_model(
        self, option: ui.RuntimeOption
    ) -> Union[ui.vision.ocr.DBDetector, ui.vision.ocr.DBCURVEDetector]:
        if self._is_curve_model:
            model = ui.vision.ocr.DBCURVEDetector(
                str(self.model_path),
                str(self.params_path),
                runtime_option=option,
            )
        else:
            model = ui.vision.ocr.DBDetector(
                str(self.model_path),
                str(self.params_path),
                runtime_option=option,
            )
        self._config_ui_preprocessor(model)
        self._config_ui_postprocessor(model)
        return model

    def _predict(self, batch_data: BatchData) -> BatchData:
        imgs = [np.ascontiguousarray(data["img"]) for data in batch_data]
        ui_results = self._ui_model.batch_predict(imgs)
        results: BatchData = []
        for data, ui_result in zip(batch_data, ui_results):
            text_det_result = self._create_text_det_result(data, ui_result)
            results.append({"result": text_det_result})
        return results

    def _config_ui_preprocessor(self, model: ui.vision.ocr.DBDetector) -> None:
        pp_config = self.config["PreProcess"]
        preprocessor = model.preprocessor
        for item in pp_config["transform_ops"]:
            op_name = next(iter(item))
            op_config = item[op_name]
            # XXX: Default values copied from
            # `paddlex.inference.models.TextDetPredictor`
            if op_name == "DecodeImage":
                if op_config["channel_first"]:
                    raise RuntimeError(
                        "`DecodeImage.channel_first` must be set to False."
                    )
            elif op_name == "DetResizeForTest":
                preprocessor.max_side_len = op_config.get("resize_long", 960)
            elif op_name == "NormalizeImage":
                if "scale" in op_config and not (
                    abs(parse_scale(op_config["scale"]) - 1 / 255) < 1e-9
                ):
                    raise RuntimeError("`NormalizeImage.scale` must be set to 1/255.")
                if "channel_num" in op_config and op_config["channel_num"] != 3:
                    raise RuntimeError("`NormalizeImage.channel_num` must be set to 3.")
                preprocessor.set_normalize(
                    op_config.get("mean", [0.485, 0.456, 0.406]),
                    op_config.get("std", [0.229, 0.224, 0.225]),
                    True,
                )
            elif op_name == "ToCHWImage":
                # Do nothing
                pass
            elif op_name == "DetLabelEncode":
                pass
            elif op_name == "KeepKeys":
                pass
            else:
                raise RuntimeError(f"Unkown preprocessing operator: {op_name}")

    def _config_ui_postprocessor(self, model: ui.vision.ocr.DBDetector) -> None:
        pp_config = self.config["PostProcess"]
        # XXX: Default values copied from
        # `paddlex.inference.models.TextDetPredictor`
        changeable_params: Dict[str, Any] = {
            "thresh": 0.3,
            "box_thresh": 0.7,
            "unclip_ratio": 2.0,
            "score_mode": "fast",
            "use_dilation": False,
        }
        unchangeable_params: Dict[str, Any] = {
            "max_candidates": 1000,
            "box_type": "quad",
        }
        if self._is_curve_model:
            changeable_params["box_type"] = unchangeable_params.pop("box_type")
        if "name" in pp_config and pp_config["name"] == "DBPostProcess":
            for name in changeable_params:
                if name in pp_config:
                    changeable_params[name] = pp_config[name]
            for name, val in unchangeable_params.items():
                if name in pp_config and pp_config[name] != val:
                    raise RuntimeError(
                        f"`DBPostProcess.{name}` must be set to {repr(val)}."
                    )
        else:
            raise RuntimeError("Invalid config")
        postprocessor = model.postprocessor
        postprocessor.det_db_thresh = changeable_params["thresh"]
        postprocessor.det_db_box_thresh = changeable_params["box_thresh"]
        postprocessor.det_db_unclip_ratio = changeable_params["unclip_ratio"]
        postprocessor.use_dilation = changeable_params["use_dilation"]
        postprocessor.det_db_score_mode = changeable_params["score_mode"]
        if self._is_curve_model:
            if changeable_params["box_type"] not in ("quad", "poly"):
                raise RuntimeError("Invalid value of `DBPostProcess.box_type`.")
            if changeable_params["box_type"] == "quad":
                postprocessor.det_db_box_type = "bbox"
            else:
                postprocessor.det_db_box_type = "poly"

    def _create_text_det_result(self, data: Data, ui_result: Any) -> TextDetResult:
        polys = [list(zip(*([iter(box)] * 2))) for box in ui_result.boxes]
        # XXX: Currently, we cannot get scores from `ui_result`, so we
        # temporarily use dummy scores here.
        dummy_scores = [0.0 for _ in ui_result.boxes]
        dic = {
            "input_path": data["input_path"],
            "dt_polys": polys,
            "dt_scores": dummy_scores,
        }
        return TextDetResult(dic)
