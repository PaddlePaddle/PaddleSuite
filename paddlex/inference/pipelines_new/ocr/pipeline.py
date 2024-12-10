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
from ..components import SortQuadBoxes, SortPolyBoxes, CropByPolys
from .result import OCRResult

########## [TODO]后续需要更新路径
from ...components.transforms import ReadImage

from ...utils.pp_option import PaddlePredictorOption
import numpy as np


class OCRPipeline(BasePipeline):
    """OCR Pipeline"""

    entities = "OCR"

    def __init__(
        self,
        config: Dict,
        device: str = None,
        pp_option: PaddlePredictorOption = None,
        use_hpip: bool = False,
        hpi_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initializes the class with given configurations and options.

        Args:
            config (Dict): Configuration dictionary containing model and other parameters.
            device (str): The device to run the prediction on. Default is None.
            pp_option (PaddlePredictorOption): Options for PaddlePaddle predictor. Default is None.
            use_hpip (bool): Whether to use high-performance inference (hpip) for prediction. Defaults to False.
            hpi_params (Optional[Dict[str, Any]]): HPIP specific parameters. Default is None.
        """
        super().__init__(
            device=device, pp_option=pp_option, use_hpip=use_hpip, hpi_params=hpi_params
        )

        text_det_model_config = config["SubModules"]["TextDetection"]
        self.text_det_model = self.create_model(text_det_model_config)

        text_rec_model_config = config["SubModules"]["TextRecognition"]
        self.text_rec_model = self.create_model(text_rec_model_config)

        self.text_type = config["text_type"]

        if self.text_type == "general":
            self._sort_boxes = SortQuadBoxes()
            self._crop_by_polys = CropByPolys(det_box_type="quad")
        elif self.text_type == "seal":
            self._sort_boxes = SortPolyBoxes()
            self._crop_by_polys = CropByPolys(det_box_type="poly")
        else:
            raise ValueError("Unsupported text type {}".format(self.text_type))

        self.img_reader = ReadImage(format="BGR")

    def predict(
        self, input: str | list[str] | np.ndarray | list[np.ndarray], **kwargs
    ) -> OCRResult:
        """Predicts OCR results for the given input.

        Args:
            input (str | list[str] | np.ndarray | list[np.ndarray]): The input image(s) or path(s) to the images.
            **kwargs: Additional keyword arguments that can be passed to the function.

        Returns:
            OCRResult: An iterable of OCRResult objects, each containing the predicted text and other relevant information.
        """
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

            ########## [TODO] Need to confirm filtering thresholds for detection and recognition modules

            dt_polys = self._sort_boxes(dt_polys)

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

                ########## [TODO] Update in the future
                for sub_img in all_subs_of_img:
                    sub_img["input"] = sub_img["img"]
                ##########

                for rec_res in self.text_rec_model(all_subs_of_img):
                    single_img_res["rec_text"].append(rec_res["rec_text"])
                    single_img_res["rec_score"].append(rec_res["rec_score"])

            yield OCRResult(single_img_res)
