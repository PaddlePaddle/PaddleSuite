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

from ..base import BasePipeline
from typing import Any, Dict, Optional
from ..components import SortQuadBoxes, CropByPolys
from .result import OCRResult

########## [TODO]后续需要更新路径
from ...components.transforms import ReadImage


class OCRPipeline(BasePipeline):
    """OCR Pipeline"""

    entities = "OCR"

    def __init__(
        self,
        config,
        device=None,
        pp_option=None,
        use_hpip: bool = False,
        hpi_params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            device=device, pp_option=pp_option, use_hpip=use_hpip, hpi_params=hpi_params
        )

        text_det_model_config = config["SubModules"]["TextDetection"]
        self.text_det_model = self.create_model(text_det_model_config)

        text_rec_model_config = config["SubModules"]["TextRecognition"]
        self.text_rec_model = self.create_model(text_rec_model_config)

        self.text_type = config["text_type"]

        self._sort_quad_boxes = SortQuadBoxes()

        if self.text_type == "common":
            self._crop_by_polys = CropByPolys(det_box_type="quad")
        elif self.text_type == "seal":
            self._crop_by_polys = CropByPolys(det_box_type="poly")
        else:
            raise ValueError("Unsupported text type {}".format(self.text_type))

        self.img_reader = ReadImage(format="BGR")

    def predict(self, input, **kwargs):
        if not isinstance(input, list):
            input_list = [input]
        else:
            input_list = input
        img_id = 1
        for input in input_list:
            if isinstance(input, str):
                image_array = next(self.img_reader(input))[0]["img"]
            else:
                image_array = input

            assert len(image_array.shape) == 3

            det_res = next(self.text_det_model(image_array))

            dt_polys = det_res["dt_polys"]
            dt_scores = det_res["dt_scores"]

            ########## [TODO]需要确认检测模块和识别模块过滤阈值等情况

            if self.text_type == "common":
                dt_polys = self._sort_quad_boxes(dt_polys)

            single_img_res = {
                "input_img": image_array,
                "dt_polys": dt_polys,
                "img_id": img_id,
                "text_type": self.text_type,
            }
            img_id += 1
            single_img_res["rec_text"] = []
            single_img_res["rec_score"] = []
            if len(dt_polys) > 0:
                all_subs_of_img = list(self._crop_by_polys(image_array, dt_polys))

                ########## [TODO]updata in future
                for sub_img in all_subs_of_img:
                    sub_img["input"] = sub_img["img"]
                ##########

                for rec_res in self.text_rec_model(all_subs_of_img):
                    single_img_res["rec_text"].append(rec_res["rec_text"])
                    single_img_res["rec_score"].append(rec_res["rec_score"])

            yield OCRResult(single_img_res)
