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

import os, sys
from typing import Any, Dict, Optional
import numpy as np
import cv2
from ..base import BasePipeline
from ..components import CropByBoxes, convert_points_to_boxes

from .result import FormulaRecognitionResult
from ...models_new.formula_recognition.result import (
    FormulaRecResult as SingleFormulaRecognitionResult,
)
from ....utils import logging
from ...utils.pp_option import PaddlePredictorOption
from ...common.reader import ReadImage
from ...common.batch_sampler import ImageBatchSampler
from ..ocr.result import OCRResult
from ..doc_preprocessor.result import DocPreprocessorResult

# [TODO] 待更新models_new到models
from ...models_new.object_detection.result import DetResult


class FormulaRecognitionPipeline(BasePipeline):
    """Formula Recognition Pipeline"""

    entities = ["formula_recognition"]

    def __init__(
        self,
        config: Dict,
        device: str = None,
        pp_option: PaddlePredictorOption = None,
        use_hpip: bool = False,
        hpi_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initializes the formula recognition pipeline.

        Args:
            config (Dict): Configuration dictionary containing various settings.
            device (str, optional): Device to run the predictions on. Defaults to None.
            pp_option (PaddlePredictorOption, optional): PaddlePredictor options. Defaults to None.
            use_hpip (bool, optional): Whether to use high-performance inference (hpip) for prediction. Defaults to False.
            hpi_params (Optional[Dict[str, Any]], optional): HPIP parameters. Defaults to None.
        """

        super().__init__(
            device=device, pp_option=pp_option, use_hpip=use_hpip, hpi_params=hpi_params
        )

        self.use_doc_preprocessor = config.get("use_doc_preprocessor", True)
        if self.use_doc_preprocessor:
            doc_preprocessor_config = config.get("SubPipelines", {}).get(
                "DocPreprocessor",
                {
                    "pipeline_config_error": "config error for doc_preprocessor_pipeline!"
                },
            )
            self.doc_preprocessor_pipeline = self.create_pipeline(
                doc_preprocessor_config
            )

        self.use_layout_detection = config.get("use_layout_detection", True)

        if self.use_layout_detection:
            layout_det_config = config.get("SubModules", {}).get(
                "LayoutDetection",
                {"model_config_error": "config error for layout_det_model!"},
            )
            self.layout_det_model = self.create_model(layout_det_config)

        formula_recognition_config = config.get("SubModules", {}).get(
            "FormulaRecognition",
            {"model_config_error": "config error for formula_rec_model!"},
        )
        self.formula_recognition_model = self.create_model(formula_recognition_config)

        self._crop_by_boxes = CropByBoxes()

        self.batch_sampler = ImageBatchSampler(batch_size=1)
        self.img_reader = ReadImage(format="BGR")

    def get_model_settings(
        self,
        use_doc_orientation_classify: Optional[bool],
        use_doc_unwarping: Optional[bool],
        use_layout_detection: Optional[bool],
    ) -> dict:
        """
        Get the model settings based on the provided parameters or default values.

        Args:
            use_doc_orientation_classify (Optional[bool]): Whether to use document orientation classification.
            use_doc_unwarping (Optional[bool]): Whether to use document unwarping.
            use_layout_detection (Optional[bool]): Whether to use layout detection.

        Returns:
            dict: A dictionary containing the model settings.
        """
        if use_doc_orientation_classify is None and use_doc_unwarping is None:
            use_doc_preprocessor = self.use_doc_preprocessor
        else:
            use_doc_preprocessor = True

        if use_layout_detection is None:
            use_layout_detection = self.use_layout_detection

        return dict(
            use_doc_preprocessor=use_doc_preprocessor,
            use_layout_detection=use_layout_detection,
        )

    def check_model_settings_valid(
        self, model_settings: Dict, layout_det_res: DetResult
    ) -> bool:
        """
        Check if the input parameters are valid based on the initialized models.

        Args:
            model_settings (Dict): A dictionary containing input parameters.
            layout_det_res (DetResult): The layout detection result.
        Returns:
            bool: True if all required models are initialized according to input parameters, False otherwise.
        """

        if model_settings["use_doc_preprocessor"] and not self.use_doc_preprocessor:
            logging.error(
                "Set use_doc_preprocessor, but the models for doc preprocessor are not initialized."
            )
            return False

        if model_settings["use_layout_detection"]:
            if layout_det_res is not None:
                logging.error(
                    "The layout detection model has already been initialized, please set use_layout_detection=False"
                )
                return False

            if not self.use_layout_detection:
                logging.error(
                    "Set use_layout_detection, but the models for layout detection are not initialized."
                )
                return False

        return True

    def predict_single_formula_recognition_res(
        self,
        image_array: np.ndarray,
    ) -> SingleFormulaRecognitionResult:
        """
        Predict formula recognition results from an image array, layout detection results.

        Args:
            image_array (np.ndarray): The input image represented as a numpy array.
            formula_box (list): The formula box coordinates.
            flag_find_nei_text (bool): Whether to find neighboring text.
        Returns:
            SingleFormulaRecognitionResult: single formula recognition result.
        """

        formula_recognition_pred = next(self.formula_recognition_model(image_array))

        return formula_recognition_pred

    def predict(
        self,
        input: str | list[str] | np.ndarray | list[np.ndarray],
        use_layout_detection: bool = True,
        use_doc_orientation_classify: bool = False,
        use_doc_unwarping: bool = False,
        layout_det_res: DetResult = None,
        **kwargs,
    ) -> FormulaRecognitionResult:
        """
        This function predicts the layout parsing result for the given input.

        Args:
            input (str | list[str] | np.ndarray | list[np.ndarray]): The input image(s) of pdf(s) to be processed.
            use_layout_detection (bool): Whether to use layout detection.
            use_doc_orientation_classify (bool): Whether to use document orientation classification.
            use_doc_unwarping (bool): Whether to use document unwarping.
            layout_det_res (DetResult): The layout detection result.
                It will be used if it is not None and use_layout_detection is False.
            **kwargs: Additional keyword arguments.

        Returns:
            formulaRecognitionResult: The predicted formula recognition result.
        """

        model_settings = self.get_model_settings(
            use_doc_orientation_classify,
            use_doc_unwarping,
            use_layout_detection,
        )

        if not self.check_model_settings_valid(model_settings, layout_det_res):
            yield {"error": "the input params for model settings are invalid!"}

        for img_id, batch_data in enumerate(self.batch_sampler(input)):
            if not isinstance(batch_data[0], str):
                # TODO: add support input_pth for ndarray and pdf
                input_path = f"{img_id}.jpg"
            else:
                input_path = batch_data[0]

            image_array = self.img_reader(batch_data)[0]

            if model_settings["use_doc_preprocessor"]:
                doc_preprocessor_res = next(
                    self.doc_preprocessor_pipeline(
                        image_array,
                        use_doc_orientation_classify=use_doc_orientation_classify,
                        use_doc_unwarping=use_doc_unwarping,
                    )
                )
            else:
                doc_preprocessor_res = {"output_img": image_array}

            doc_preprocessor_image = doc_preprocessor_res["output_img"]

            formula_res_list = []
            formula_region_id = 1

            if not model_settings["use_layout_detection"] and layout_det_res is None:
                layout_det_res = {}
                img_height, img_width = doc_preprocessor_image.shape[:2]
                single_formula_rec_res = self.predict_single_formula_recognition_res(
                    doc_preprocessor_image,
                )
                single_formula_rec_res["formula_region_id"] = formula_region_id
                formula_res_list.append(single_formula_rec_res)
                formula_region_id += 1
            else:
                if model_settings["use_layout_detection"]:
                    layout_det_res = next(self.layout_det_model(doc_preprocessor_image))
                for box_info in layout_det_res["boxes"]:
                    if box_info["label"].lower() in ["formula"]:
                        crop_img_info = self._crop_by_boxes(image_array, [box_info])
                        crop_img_info = crop_img_info[0]
                        single_formula_rec_res = (
                            self.predict_single_formula_recognition_res(
                                crop_img_info["img"]
                            )
                        )
                        single_formula_rec_res["formula_region_id"] = formula_region_id
                        single_formula_rec_res["dt_polys"] = box_info["coordinate"]
                        formula_res_list.append(single_formula_rec_res)
                        formula_region_id += 1

            single_img_res = {
                "input_path": input_path,
                "layout_det_res": layout_det_res,
                "doc_preprocessor_res": doc_preprocessor_res,
                "formula_res_list": formula_res_list,
                "model_settings": model_settings,
            }
            yield FormulaRecognitionResult(single_img_res)
