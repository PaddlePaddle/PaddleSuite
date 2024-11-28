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
from scipy.ndimage import rotate
from .result import DocPreprocessorResult

########## [TODO]后续需要更新路径
from ...components.transforms import ReadImage


class DocPreprocessorPipeline(BasePipeline):
    """Doc Preprocessor Pipeline"""

    entities = "doc_preprocessor"

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

        self.use_doc_orientation_classify = True
        if "use_doc_orientation_classify" in config:
            self.use_doc_orientation_classify = config["use_doc_orientation_classify"]

        self.use_doc_unwarping = True
        if "use_doc_unwarping" in config:
            self.use_doc_unwarping = config["use_doc_unwarping"]

        if self.use_doc_orientation_classify:
            doc_ori_classify_config = config["SubModules"]["DocOrientationClassify"]
            self.doc_ori_classify_model = self.create_model(doc_ori_classify_config)

        if self.use_doc_unwarping:
            doc_unwarping_config = config["SubModules"]["DocUnwarping"]
            self.doc_unwarping_model = self.create_model(doc_unwarping_config)

        self.img_reader = ReadImage(format="BGR")

    def rotate_image(self, image_array, rotate_angle):
        """rotate image"""
        assert (
            rotate_angle >= 0 and rotate_angle < 360
        ), "rotate_angle must in [0-360), but get {rotate_angle}."
        return rotate(image_array, rotate_angle, reshape=True)

    def check_input_params(self, input_params):

        if (
            input_params["use_doc_orientation_classify"]
            and not self.use_doc_orientation_classify
        ):
            raise ValueError(
                "The model for doc orientation classify is not initialized."
            )

        if input_params["use_doc_unwarping"] and not self.use_doc_unwarping:
            raise ValueError("The model for doc unwarping is not initialized.")

        return

    def predict(
        self,
        input,
        use_doc_orientation_classify=True,
        use_doc_unwarping=False,
        **kwargs
    ):

        if not isinstance(input, list):
            input_list = [input]
        else:
            input_list = input

        input_params = {
            "use_doc_orientation_classify": use_doc_orientation_classify,
            "use_doc_unwarping": use_doc_unwarping,
        }
        self.check_input_params(input_params)

        img_id = 1
        for input in input_list:
            if isinstance(input, str):
                image_array = next(self.img_reader(input))[0]["img"]
            else:
                image_array = input

            assert len(image_array.shape) == 3

            if input_params["use_doc_orientation_classify"]:
                pred = next(self.doc_ori_classify_model(image_array))
                angle = int(pred["label_names"][0])
                rot_img = self.rotate_image(image_array, angle)
            else:
                angle = -1
                rot_img = image_array

            if input_params["use_doc_unwarping"]:
                output_img = next(self.doc_unwarping_model(rot_img))["doctr_img"]
            else:
                output_img = rot_img

            single_img_res = {
                "input_image": image_array,
                "input_params": input_params,
                "angle": angle,
                "rot_img": rot_img,
                "output_img": output_img,
                "img_id": img_id,
            }
            img_id += 1
            yield DocPreprocessorResult(single_img_res)
