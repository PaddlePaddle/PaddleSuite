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

from pathlib import Path
import numpy as np

from ..utils.io import ImageReader
from ..components import CropByBoxes
from ..results import AttributeRecResult
from .base import BasePipeline


class AttributeRecPipeline(BasePipeline):
    """Attribute Rec Pipeline"""

    def __init__(
        self,
        det_model,
        cls_model,
        det_batch_size=1,
        cls_batch_size=1,
        device=None,
        predictor_kwargs=None,
    ):
        super().__init__(device, predictor_kwargs)
        self._build_predictor(det_model, cls_model)
        self.set_predictor(det_batch_size, cls_batch_size, device)

    def _build_predictor(self, det_model, cls_model):
        self.det_model = self._create(model=det_model)
        self.cls_model = self._create(model=cls_model)
        self._crop_by_boxes = CropByBoxes()
        self._img_reader = ImageReader(backend="opencv")

    def set_predictor(self, det_batch_size=None, cls_batch_size=None, device=None):
        if det_batch_size:
            self.det_model.set_predictor(batch_size=det_batch_size)
        if cls_batch_size:
            self.cls_model.set_predictor(batch_size=cls_batch_size)
        if device:
            self.det_model.set_predictor(device=device)
            self.cls_model.set_predictor(device=device)

    def predict(self, input, **kwargs):
        self.set_predictor(**kwargs)
        for det_res in self.det_model(input):
            cls_res = self.get_cls_result(det_res)
            yield self.get_final_result(det_res, cls_res)

    def get_cls_result(self, det_res):
        subs_of_img = list(self._crop_by_boxes(det_res))
        img_list = [img["img"] for img in subs_of_img]
        all_cls_res = list(self.cls_model(img_list))
        output = {"label": [], "score": []}
        for res in all_cls_res:
            output["label"].append(res["label_names"])
            output["score"].append(res["scores"])
        return output

    def get_final_result(self, det_res, cls_res):
        single_img_res = {"input_path": det_res["input_path"], "boxes": []}
        for i, obj in enumerate(det_res["boxes"]):
            cls_scores = cls_res["score"][i]
            labels = cls_res["label"][i]
            single_img_res["boxes"].append(
                {
                    "labels": labels,
                    "cls_scores": cls_scores,
                    "det_score": obj["score"],
                    "coordinate": obj["coordinate"],
                }
            )
        return AttributeRecResult(single_img_res)


class PedestrianAttributeRecPipeline(AttributeRecPipeline):
    entities = "pedestrian_attribute_recognition"


class VehicleAttributeRecPipeline(AttributeRecPipeline):
    entities = "vehicle_attribute_recognition"
