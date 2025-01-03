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
from ..components import CropByBoxes
from ..layout_parsing.utils import convert_points_to_boxes
from .utils import get_neighbor_boxes_idx
from .table_recognition_post_processing import get_table_recognition_res
from .result import SingleTableRecognitionResult, TableRecognitionResult
from ....utils import logging
from ...utils.pp_option import PaddlePredictorOption
from ...common.reader import ReadImage
from ...common.batch_sampler import ImageBatchSampler
from ..ocr.result import OCRResult
from ..doc_preprocessor.result import DocPreprocessorResult

# [TODO] 待更新models_new到models
from ...models_new.object_detection.result import DetResult


class TableRecognitionPipeline(BasePipeline):
    """Table Recognition Pipeline"""

    entities = ["table_recognition"]

    def __init__(
        self,
        config: Dict,
        device: str = None,
        pp_option: PaddlePredictorOption = None,
        use_hpip: bool = False,
        hpi_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initializes the layout parsing pipeline.

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

        self.use_doc_preprocessor = False
        if "use_doc_preprocessor" in config:
            self.use_doc_preprocessor = config["use_doc_preprocessor"]

        if self.use_doc_preprocessor:
            doc_preprocessor_config = config["SubPipelines"]["DocPreprocessor"]
            self.doc_preprocessor_pipeline = self.create_pipeline(
                doc_preprocessor_config
            )

        self.use_layout_detection = True
        if "use_layout_detection" in config:
            self.use_layout_detection = config["use_layout_detection"]

        if self.use_layout_detection:
            layout_det_config = config["SubModules"]["LayoutDetection"]
            self.layout_det_model = self.create_model(layout_det_config)

        table_structure_config = config["SubModules"]["TableStructureRecognition"]
        self.table_structure_model = self.create_model(table_structure_config)

        self.use_ocr_model = True
        if "use_ocr_model" in config:
            self.use_ocr_model = config["use_ocr_model"]
        if self.use_ocr_model:
            general_ocr_config = config["SubPipelines"]["GeneralOCR"]
            self.general_ocr_pipeline = self.create_pipeline(general_ocr_config)

        self._crop_by_boxes = CropByBoxes()

        self.batch_sampler = ImageBatchSampler(batch_size=1)
        self.img_reader = ReadImage(format="BGR")

    def check_input_params_valid(
        self, input_params: Dict, overall_ocr_res: OCRResult, layout_det_res: DetResult
    ) -> bool:
        """
        Check if the input parameters are valid based on the initialized models.

        Args:
            input_params (Dict): A dictionary containing input parameters.
            overall_ocr_res (OCRResult): Overall OCR result obtained after running the OCR pipeline.
                The overall OCR result with convert_points_to_boxes information.
            layout_det_res (DetResult): The layout detection result.
        Returns:
            bool: True if all required models are initialized according to input parameters, False otherwise.
        """

        if input_params["use_doc_preprocessor"] and not self.use_doc_preprocessor:
            logging.error(
                "Set use_doc_preprocessor, but the models for doc preprocessor are not initialized."
            )
            return False

        if input_params["use_layout_detection"]:
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

        if input_params["use_ocr_model"]:
            if overall_ocr_res is not None:
                logging.error(
                    "The OCR models have already been initialized, please set use_ocr_model=False"
                )
                return False

            if not self.use_ocr_model:
                logging.error(
                    "Set use_ocr_model, but the models for OCR are not initialized."
                )
                return False

        return True

    def predict_doc_preprocessor_res(
        self, image_array: np.ndarray, input_params: dict
    ) -> tuple[DocPreprocessorResult, np.ndarray]:
        """
        Preprocess the document image based on input parameters.

        Args:
            image_array (np.ndarray): The input image array.
            input_params (dict): Dictionary containing preprocessing parameters.

        Returns:
            tuple[DocPreprocessorResult, np.ndarray]: A tuple containing the preprocessing
                                              result dictionary and the processed image array.
        """
        if input_params["use_doc_preprocessor"]:
            use_doc_orientation_classify = input_params["use_doc_orientation_classify"]
            use_doc_unwarping = input_params["use_doc_unwarping"]
            doc_preprocessor_res = next(
                self.doc_preprocessor_pipeline(
                    image_array,
                    use_doc_orientation_classify=use_doc_orientation_classify,
                    use_doc_unwarping=use_doc_unwarping,
                )
            )
            doc_preprocessor_image = doc_preprocessor_res["output_img"]
        else:
            doc_preprocessor_res = {}
            doc_preprocessor_image = image_array
        return doc_preprocessor_res, doc_preprocessor_image

    def predict_single_table_recognition_res(
        self,
        image_array: np.ndarray,
        overall_ocr_res: OCRResult,
        table_box: list,
        flag_find_nei_text: bool = True,
    ) -> SingleTableRecognitionResult:
        """
        Predict table recognition results from an image array, layout detection results, and OCR results.

        Args:
            image_array (np.ndarray): The input image represented as a numpy array.
            overall_ocr_res (OCRResult): Overall OCR result obtained after running the OCR pipeline.
                The overall OCR results containing text recognition information.
            table_box (list): The table box coordinates.
            flag_find_nei_text (bool): Whether to find neighboring text.
        Returns:
            SingleTableRecognitionResult: single table recognition result.
        """
        table_structure_pred = next(self.table_structure_model(image_array))
        single_table_recognition_res = get_table_recognition_res(
            table_box, table_structure_pred, overall_ocr_res
        )
        neighbor_text = ""
        if flag_find_nei_text:
            match_idx_list = get_neighbor_boxes_idx(
                overall_ocr_res["dt_boxes"], table_box
            )
            if len(match_idx_list) > 0:
                for idx in match_idx_list:
                    neighbor_text += overall_ocr_res["rec_text"][idx] + "; "
        single_table_recognition_res["neighbor_text"] = neighbor_text
        return single_table_recognition_res

    def predict(
        self,
        input: str | list[str] | np.ndarray | list[np.ndarray],
        use_layout_detection: bool = True,
        use_doc_orientation_classify: bool = False,
        use_doc_unwarping: bool = False,
        overall_ocr_res: OCRResult = None,
        layout_det_res: DetResult = None,
        **kwargs
    ) -> TableRecognitionResult:
        """
        This function predicts the layout parsing result for the given input.

        Args:
            input (str | list[str] | np.ndarray | list[np.ndarray]): The input image(s) of pdf(s) to be processed.
            use_layout_detection (bool): Whether to use layout detection.
            use_doc_orientation_classify (bool): Whether to use document orientation classification.
            use_doc_unwarping (bool): Whether to use document unwarping.
            overall_ocr_res (OCRResult): The overall OCR result with convert_points_to_boxes information.
                It will be used if it is not None and use_ocr_model is False.
            layout_det_res (DetResult): The layout detection result.
                It will be used if it is not None and use_layout_detection is False.
            **kwargs: Additional keyword arguments.

        Returns:
            TableRecognitionResult: The predicted table recognition result.
        """

        input_params = {
            "use_layout_detection": use_layout_detection,
            "use_doc_preprocessor": self.use_doc_preprocessor,
            "use_doc_orientation_classify": use_doc_orientation_classify,
            "use_doc_unwarping": use_doc_unwarping,
            "use_ocr_model": self.use_ocr_model,
        }

        if use_doc_orientation_classify or use_doc_unwarping:
            input_params["use_doc_preprocessor"] = True
        else:
            input_params["use_doc_preprocessor"] = False

        if not self.check_input_params_valid(
            input_params, overall_ocr_res, layout_det_res
        ):
            yield None

        for img_id, batch_data in enumerate(self.batch_sampler(input)):
            image_array = self.img_reader(batch_data)[0]
            img_id += 1

            doc_preprocessor_res, doc_preprocessor_image = (
                self.predict_doc_preprocessor_res(image_array, input_params)
            )

            if self.use_ocr_model:
                overall_ocr_res = next(
                    self.general_ocr_pipeline(doc_preprocessor_image)
                )
                dt_boxes = convert_points_to_boxes(overall_ocr_res["dt_polys"])
                overall_ocr_res["dt_boxes"] = dt_boxes

            table_res_list = []
            table_region_id = 1
            if not input_params["use_layout_detection"] and layout_det_res is None:
                layout_det_res = {}
                img_height, img_width = doc_preprocessor_image.shape[:2]
                table_box = [0, 0, img_width - 1, img_height - 1]
                single_table_rec_res = self.predict_single_table_recognition_res(
                    doc_preprocessor_image,
                    overall_ocr_res,
                    table_box,
                    flag_find_nei_text=False,
                )
                single_table_rec_res["table_region_id"] = table_region_id
                table_res_list.append(single_table_rec_res)
                table_region_id += 1
            else:
                if input_params["use_layout_detection"]:
                    layout_det_res = next(self.layout_det_model(doc_preprocessor_image))
                for box_info in layout_det_res["boxes"]:
                    if box_info["label"].lower() in ["table"]:
                        crop_img_info = self._crop_by_boxes(image_array, [box_info])
                        crop_img_info = crop_img_info[0]
                        table_box = crop_img_info["box"]
                        single_table_rec_res = (
                            self.predict_single_table_recognition_res(
                                crop_img_info["img"], overall_ocr_res, table_box
                            )
                        )
                        single_table_rec_res["table_region_id"] = table_region_id
                        table_res_list.append(single_table_rec_res)
                        table_region_id += 1

            single_img_res = {
                "layout_det_res": layout_det_res,
                "doc_preprocessor_res": doc_preprocessor_res,
                "overall_ocr_res": overall_ocr_res,
                "table_res_list": table_res_list,
                "input_params": input_params,
                "img_id": img_id,
            }
            yield TableRecognitionResult(single_img_res)
