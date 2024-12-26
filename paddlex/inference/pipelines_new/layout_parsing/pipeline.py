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
import numpy as np
import cv2
from ..components import CropByBoxes
from .utils import (
    convert_points_to_boxes,
    get_sub_regions_ocr_res,
    get_neighbor_boxes_idx,
)
from .table_recognition_post_processing import get_table_recognition_res

from .result import LayoutParsingResult, TableRecognitionResult

from ....utils import logging

from ...utils.pp_option import PaddlePredictorOption

from ...common.reader import ReadImage
from ...common.batch_sampler import ImageBatchSampler

from ..ocr.result import OCRResult
from ..doc_preprocessor.result import DocPreprocessorResult

# [TODO] 待更新models_new到models
from ...models_new.object_detection.result import DetResult

import os, sys


class LayoutParsingPipeline(BasePipeline):
    """Layout Parsing Pipeline"""

    entities = ["layout_parsing", "seal_recognition", "table_recognition"]

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

        self.inintial_predictor(config)

        self.batch_sampler = ImageBatchSampler(batch_size=1)
        self.img_reader = ReadImage(format="BGR")

        self._crop_by_boxes = CropByBoxes()

    def set_used_models_flag(self, config: Dict) -> None:
        """
        Set the flags for which models to use based on the configuration.

        Args:
            config (Dict): A dictionary containing configuration settings.

        Returns:
            None
        """
        pipeline_name = config["pipeline_name"]

        self.pipeline_name = pipeline_name

        self.use_layout_detection = True
        self.use_doc_preprocessor = False
        self.use_general_ocr = False
        self.use_seal_recognition = False
        self.use_table_recognition = False

        if "use_layout_detection" in config:
            self.use_layout_detection = config["use_layout_detection"]

        if "use_doc_preprocessor" in config:
            self.use_doc_preprocessor = config["use_doc_preprocessor"]

        if pipeline_name == "layout_parsing":
            if "use_general_ocr" in config:
                self.use_general_ocr = config["use_general_ocr"]
            if "use_seal_recognition" in config:
                self.use_seal_recognition = config["use_seal_recognition"]
            if "use_table_recognition" in config:
                self.use_table_recognition = config["use_table_recognition"]

        elif pipeline_name == "seal_recognition":
            self.use_seal_recognition = True

        elif pipeline_name == "table_recognition":
            self.use_table_recognition = True

    def inintial_predictor(self, config: Dict) -> None:
        """Initializes the predictor based on the provided configuration.

        Args:
            config (Dict): A dictionary containing the configuration for the predictor.

        Returns:
            None
        """

        self.set_used_models_flag(config)

        if self.use_layout_detection:
            layout_det_config = config["SubModules"]["LayoutDetection"]
            self.layout_det_model = self.create_model(layout_det_config)

        if self.use_doc_preprocessor:
            doc_preprocessor_config = config["SubPipelines"]["DocPreprocessor"]
            self.doc_preprocessor_pipeline = self.create_pipeline(
                doc_preprocessor_config
            )

        if self.use_general_ocr:
            general_ocr_config = config["SubPipelines"]["GeneralOCR"]
            self.general_ocr_pipeline = self.create_pipeline(general_ocr_config)

        if self.use_seal_recognition:
            seal_ocr_config = config["SubPipelines"]["SealOCR"]
            self.seal_ocr_pipeline = self.create_pipeline(seal_ocr_config)

        if self.use_table_recognition:
            table_structure_config = config["SubModules"]["TableStructureRecognition"]
            self.table_structure_model = self.create_model(table_structure_config)
            if not self.use_general_ocr:
                general_ocr_config = config["SubPipelines"]["GeneralOCR"]
                self.general_ocr_pipeline = self.create_pipeline(general_ocr_config)
        return

    def get_text_paragraphs_ocr_res(
        self, overall_ocr_res: OCRResult, layout_det_res: DetResult
    ) -> OCRResult:
        """
        Retrieves the OCR results for text paragraphs, excluding those of formulas, tables, and seals.

        Args:
            overall_ocr_res (OCRResult): The overall OCR result containing text information.
            layout_det_res (DetResult): The detection result containing the layout information of the document.

        Returns:
            OCRResult: The OCR result for text paragraphs after excluding formulas, tables, and seals.
        """
        object_boxes = []
        for box_info in layout_det_res["boxes"]:
            if box_info["label"].lower() in ["formula", "table", "seal"]:
                object_boxes.append(box_info["coordinate"])
        object_boxes = np.array(object_boxes)
        return get_sub_regions_ocr_res(overall_ocr_res, object_boxes, flag_within=False)

    def check_input_params_valid(self, input_params: Dict) -> bool:
        """
        Check if the input parameters are valid based on the initialized models.

        Args:
            input_params (Dict): A dictionary containing input parameters.

        Returns:
            bool: True if all required models are initialized according to input parameters, False otherwise.
        """

        if input_params["use_doc_preprocessor"] and not self.use_doc_preprocessor:
            logging.error(
                "Set use_doc_preprocessor, but the models for doc preprocessor are not initialized."
            )
            return False

        if input_params["use_general_ocr"] and not self.use_general_ocr:
            logging.error(
                "Set use_general_ocr, but the models for general OCR are not initialized."
            )
            return False

        if input_params["use_seal_recognition"] and not self.use_seal_recognition:
            logging.error(
                "Set use_seal_recognition, but the models for seal recognition are not initialized."
            )
            return False

        if input_params["use_table_recognition"] and not self.use_table_recognition:
            logging.error(
                "Set use_table_recognition, but the models for table recognition are not initialized."
            )
            return False

        return True

    def convert_input_params(self, input_params: Dict) -> None:
        """
        Convert input parameters based on the pipeline name.

        Args:
            input_params (Dict): The input parameters dictionary.

        Returns:
            None
        """
        if self.pipeline_name == "seal_recognition":
            input_params["use_general_ocr"] = False
            input_params["use_table_recognition"] = False
        elif self.pipeline_name == "table_recognition":
            input_params["use_general_ocr"] = False
            input_params["use_seal_recognition"] = False
        return

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

    def predict_layout_detection_res(
        self, image_array: np.ndarray, input_params: dict
    ) -> DetResult:
        """Predict layout detection result based on input parameters.

        Args:
            image_array (np.ndarray): The input image array.
            input_params (dict): The input parameters including use_layout_detection, use_table_recognition, and use_seal_recognition.

        Returns:
            DetResult: The predicted layout detection result.
        """
        if input_params["use_layout_detection"]:
            ########## [TODO]RT-DETR 检测结果有重复
            layout_det_res = next(self.layout_det_model(image_array))
            layout_det_res["layout_det_use_whole_image"] = False
        else:
            layout_det_res = {}
            layout_det_res["layout_det_use_whole_image"] = True
            layout_det_res["boxes"] = []
            img_height, img_width = image_array.shape[:2]
            coordinate = [0, 0, img_width, img_height]
            if input_params["use_table_recognition"]:
                box_info = {"label": "table", "coordinate": coordinate}
                layout_det_res["boxes"].append(box_info)
            if input_params["use_seal_recognition"]:
                box_info = {"label": "seal", "coordinate": coordinate}
                layout_det_res["boxes"].append(box_info)
        return layout_det_res

    def predict_overall_ocr_res(self, image_array: np.ndarray) -> OCRResult:
        """
        Predict the overall OCR result for the given image array.

        Args:
            image_array (np.ndarray): The input image array to perform OCR on.

        Returns:
            OCRResult: The predicted OCR result with updated dt_boxes.
        """
        overall_ocr_res = next(self.general_ocr_pipeline(image_array))
        dt_boxes = convert_points_to_boxes(overall_ocr_res["dt_polys"])
        overall_ocr_res["dt_boxes"] = dt_boxes
        return overall_ocr_res

    def predict_table_recognition_res(
        self,
        image_array: np.ndarray,
        layout_det_res: DetResult,
        overall_ocr_res: OCRResult,
    ) -> list[TableRecognitionResult]:
        """
        Predict table recognition results from an image array, layout detection results, and OCR results.

        Args:
            image_array (np.ndarray): The input image represented as a numpy array.
            layout_det_res (DetResult): The layout detection results containing box information.
            overall_ocr_res (OCRResult): The overall OCR results containing text recognition information.

        Returns:
            list[TableRecognitionResult]: A list of table recognition results.
        """
        table_res_list = []
        table_region_id = 1
        for box_info in layout_det_res["boxes"]:
            if box_info["label"].lower() in ["table"]:
                if layout_det_res["layout_det_use_whole_image"]:
                    crop_img_info = {}
                    crop_img_info["img"] = image_array
                    crop_img_info["box"] = box_info["coordinate"]
                else:
                    crop_img_info = self._crop_by_boxes(image_array, [box_info])
                    crop_img_info = crop_img_info[0]
                table_structure_pred = next(
                    self.table_structure_model(crop_img_info["img"])
                )
                table_recognition_res = get_table_recognition_res(
                    crop_img_info, table_structure_pred, overall_ocr_res
                )

                neighbor_text = ""
                match_idx_list = get_neighbor_boxes_idx(
                    overall_ocr_res["dt_boxes"], box_info["coordinate"]
                )
                if len(match_idx_list) > 0:
                    for idx in match_idx_list:
                        neighbor_text += overall_ocr_res["rec_text"][idx] + "; "
                table_recognition_res["neighbor_text"] = neighbor_text
                table_recognition_res["table_region_id"] = table_region_id
                table_region_id += 1
                table_res_list.append(table_recognition_res)
        return table_res_list

    def predict_seal_recognition_res(
        self, image_array: np.ndarray, layout_det_res: DetResult
    ) -> list[OCRResult]:
        """
        Predict seal recognition results based on the input image and layout detection results.

        Args:
            image_array (np.ndarray): The input image represented as a NumPy array.
            layout_det_res (DetResult): The layout detection results containing boxes and labels.

        Returns:
            list[OCRResult]: A list of OCR results for detected seal regions.
        """
        seal_res_list = []
        seal_region_id = 1
        for box_info in layout_det_res["boxes"]:
            if box_info["label"].lower() in ["seal"]:
                if layout_det_res["layout_det_use_whole_image"]:
                    crop_img_info = {}
                    crop_img_info["img"] = image_array
                    crop_img_info["box"] = box_info["coordinate"]
                else:
                    crop_img_info = self._crop_by_boxes(image_array, [box_info])
                    crop_img_info = crop_img_info[0]
                seal_ocr_res = next(self.seal_ocr_pipeline(crop_img_info["img"]))
                seal_ocr_res["seal_region_id"] = seal_region_id
                seal_region_id += 1
                seal_res_list.append(seal_ocr_res)
        return seal_res_list

    def predict(
        self,
        input: str | list[str] | np.ndarray | list[np.ndarray],
        use_layout_detection: bool = True,
        use_doc_orientation_classify: bool = False,
        use_doc_unwarping: bool = False,
        use_general_ocr: bool = True,
        use_seal_recognition: bool = True,
        use_table_recognition: bool = True,
        **kwargs
    ) -> LayoutParsingResult:
        """
        This function predicts the layout parsing result for the given input.

        Args:
            input (str | list[str] | np.ndarray | list[np.ndarray]): The input image(s) of pdf(s) to be processed.
            use_layout_detection (bool): Whether to use layout detection.
            use_doc_orientation_classify (bool): Whether to use document orientation classification.
            use_doc_unwarping (bool): Whether to use document unwarping.
            use_general_ocr (bool): Whether to use general OCR.
            use_seal_recognition (bool): Whether to use seal recognition.
            use_table_recognition (bool): Whether to use table recognition.
            **kwargs: Additional keyword arguments.

        Returns:
            LayoutParsingResult: The predicted layout parsing result.
        """

        input_params = {
            "use_layout_detection": use_layout_detection,
            "use_doc_preprocessor": self.use_doc_preprocessor,
            "use_doc_orientation_classify": use_doc_orientation_classify,
            "use_doc_unwarping": use_doc_unwarping,
            "use_general_ocr": use_general_ocr,
            "use_seal_recognition": use_seal_recognition,
            "use_table_recognition": use_table_recognition,
        }

        self.convert_input_params(input_params)

        if use_doc_orientation_classify or use_doc_unwarping:
            input_params["use_doc_preprocessor"] = True
        else:
            input_params["use_doc_preprocessor"] = False

        if not self.check_input_params_valid(input_params):
            yield {"error": "input params invalid"}

        for img_id, batch_data in enumerate(self.batch_sampler(input)):
            image_array = self.img_reader(batch_data)[0]
            img_id += 1

            doc_preprocessor_res, doc_preprocessor_image = (
                self.predict_doc_preprocessor_res(image_array, input_params)
            )

            layout_det_res = self.predict_layout_detection_res(
                doc_preprocessor_image, input_params
            )

            if input_params["use_general_ocr"] or input_params["use_table_recognition"]:
                overall_ocr_res = self.predict_overall_ocr_res(doc_preprocessor_image)
            else:
                overall_ocr_res = {}

            if input_params["use_general_ocr"]:
                text_paragraphs_ocr_res = self.get_text_paragraphs_ocr_res(
                    overall_ocr_res, layout_det_res
                )
            else:
                text_paragraphs_ocr_res = {}

            if input_params["use_table_recognition"]:
                table_res_list = self.predict_table_recognition_res(
                    doc_preprocessor_image, layout_det_res, overall_ocr_res
                )
            else:
                table_res_list = []

            if input_params["use_seal_recognition"]:
                seal_res_list = self.predict_seal_recognition_res(
                    doc_preprocessor_image, layout_det_res
                )
            else:
                seal_res_list = []

            single_img_res = {
                "layout_det_res": layout_det_res,
                "doc_preprocessor_res": doc_preprocessor_res,
                "overall_ocr_res": overall_ocr_res,
                "text_paragraphs_ocr_res": text_paragraphs_ocr_res,
                "table_res_list": table_res_list,
                "seal_res_list": seal_res_list,
                "input_params": input_params,
                "img_id": img_id,
            }
            yield LayoutParsingResult(single_img_res)
