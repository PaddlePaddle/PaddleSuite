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
from typing import List

import ultrainfer as ui
import numpy as np
from paddlex.inference.results import TextRecResult
from paddlex.modules.text_recognition.model_list import MODELS

from paddlex_hpi._utils.typing import BatchData, Data
from paddlex_hpi.models.base import CVPredictor


class TextRecPredictor(CVPredictor):
    entities = MODELS

    def _build_ui_model(self, option: ui.RuntimeOption) -> ui.vision.ocr.Recognizer:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt") as f:
            pp_config = self.config["PostProcess"]
            for lab in pp_config["character_dict"]:
                f.write(lab + "\n")
            f.flush()
            model = ui.vision.ocr.Recognizer(
                str(self.model_path),
                str(self.params_path),
                label_path=f.name,
                runtime_option=option,
            )
            self._config_ui_preprocessor(model)
        return model

    def _predict(self, batch_data: BatchData) -> BatchData:
        imgs = [np.ascontiguousarray(data["img"]) for data in batch_data]
        ui_result = self._ui_model.batch_predict(imgs)
        results: BatchData = []
        for data, text, score in zip(batch_data, ui_result.text, ui_result.rec_scores):
            text_rec_result = self._create_text_rec_result(data, text, score)
            results.append({"result": text_rec_result})
        return results

    def _config_ui_preprocessor(self, model: ui.vision.ocr.Recognizer) -> None:
        pp_config = self.config["PreProcess"]
        preprocessor = model.preprocessor
        found_resize_op = False
        for item in pp_config["transform_ops"]:
            op_name = next(iter(item))
            op_config = item[op_name]
            if op_name == "DecodeImage":
                if op_config["channel_first"]:
                    raise RuntimeError(
                        "`DecodeImage.channel_first` must be set to False."
                    )
            elif op_name == "RecResizeImg":
                preprocessor.rec_image_shape = op_config["image_shape"]
                found_resize_op = True
            elif op_name == "MultiLabelEncode":
                pass
            elif op_name == "KeepKeys":
                pass
            else:
                raise RuntimeError(f"Unkown preprocessing operator: {op_name}")
        if not found_resize_op:
            raise RuntimeError("Could not find the config for `RecResizeImg`.")

    def _create_text_rec_result(
        self, data: Data, text: str, score: float
    ) -> TextRecResult:
        dic = {
            "input_path": data["input_path"],
            "rec_text": text,
            "rec_score": score,
        }
        return TextRecResult(dic)
