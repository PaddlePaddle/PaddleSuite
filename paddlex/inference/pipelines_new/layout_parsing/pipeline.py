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
from .utils import convert_points_to_boxes, get_sub_regions_ocr_res
from .table_recognition_post_processing import get_table_recognition_res

from .result import LayoutParsingResult

from ....utils import logging

from ...utils.pp_option import PaddlePredictorOption

########## [TODO]后续需要更新路径
from ...components.transforms import ReadImage

from ..ocr.result import OCRResult
from ...results import DetResult


class LayoutParsingPipeline(BasePipeline):
    """Layout Parsing Pipeline"""

    entities = "layout_parsing"

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

        self.img_reader = ReadImage(format="BGR")

        self._crop_by_boxes = CropByBoxes()

    def inintial_predictor(self, config: Dict) -> None:
        """Initializes the predictor based on the provided configuration.

        Args:
            config (Dict): A dictionary containing the configuration for the predictor.

        Returns:
            None
        """

        layout_det_config = config["SubModules"]["LayoutDetection"]
        self.layout_det_model = self.create_model(layout_det_config)

        self.use_doc_preprocessor = False
        if "use_doc_preprocessor" in config:
            self.use_doc_preprocessor = config["use_doc_preprocessor"]

        if self.use_doc_preprocessor:
            doc_preprocessor_config = config["SubPipelines"]["DocPreprocessor"]
            self.doc_preprocessor_pipeline = self.create_pipeline(
                doc_preprocessor_config
            )

        self.use_common_ocr = False
        if "use_common_ocr" in config:
            self.use_common_ocr = config["use_common_ocr"]
        if self.use_common_ocr:
            common_ocr_config = config["SubPipelines"]["CommonOCR"]
            self.common_ocr_pipeline = self.create_pipeline(common_ocr_config)

        self.use_seal_recognition = False
        if "use_seal_recognition" in config:
            self.use_seal_recognition = config["use_seal_recognition"]
        if self.use_seal_recognition:
            seal_ocr_config = config["SubPipelines"]["SealOCR"]
            self.seal_ocr_pipeline = self.create_pipeline(seal_ocr_config)

        self.use_table_recognition = False
        if "use_table_recognition" in config:
            self.use_table_recognition = config["use_table_recognition"]
        if self.use_table_recognition:
            table_structure_config = config["SubModules"]["TableStructurePredictor"]
            self.table_structure_model = self.create_model(table_structure_config)
            if not self.use_common_ocr:
                common_ocr_config = config["SubPipelines"]["OCR"]
                self.common_ocr_pipeline = self.create_pipeline(common_ocr_config)
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

        if input_params["use_common_ocr"] and not self.use_common_ocr:
            logging.error(
                "Set use_common_ocr, but the models for common OCR are not initialized."
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

    def predict(
        self,
        input: str | list[str] | np.ndarray | list[np.ndarray],
        use_doc_orientation_classify: bool = False,
        use_doc_unwarping: bool = False,
        use_common_ocr: bool = True,
        use_seal_recognition: bool = True,
        use_table_recognition: bool = True,
        **kwargs
    ) -> LayoutParsingResult:
        """
        This function predicts the layout parsing result for the given input.

        Args:
            input (str | list[str] | np.ndarray | list[np.ndarray]): The input image(s) to be processed.
            use_doc_orientation_classify (bool): Whether to use document orientation classification.
            use_doc_unwarping (bool): Whether to use document unwarping.
            use_common_ocr (bool): Whether to use common OCR.
            use_seal_recognition (bool): Whether to use seal recognition.
            use_table_recognition (bool): Whether to use table recognition.
            **kwargs: Additional keyword arguments.

        Returns:
            LayoutParsingResult: The predicted layout parsing result.
        """

        if not isinstance(input, list):
            input_list = [input]
        else:
            input_list = input

        input_params = {
            "use_doc_preprocessor": self.use_doc_preprocessor,
            "use_doc_orientation_classify": use_doc_orientation_classify,
            "use_doc_unwarping": use_doc_unwarping,
            "use_common_ocr": use_common_ocr,
            "use_seal_recognition": use_seal_recognition,
            "use_table_recognition": use_table_recognition,
        }

        if use_doc_orientation_classify or use_doc_unwarping:
            input_params["use_doc_preprocessor"] = True
        else:
            input_params["use_doc_preprocessor"] = False

        if not self.check_input_params_valid(input_params):
            yield {"error": "input params invalid"}

        img_id = 1
        for input in input_list:
            if isinstance(input, str):
                image_array = next(self.img_reader(input))[0]["img"]
            else:
                image_array = input

            assert len(image_array.shape) == 3

            if input_params["use_doc_preprocessor"]:
                doc_preprocessor_res = next(
                    self.doc_preprocessor_pipeline(
                        image_array,
                        use_doc_orientation_classify=use_doc_orientation_classify,
                        use_doc_unwarping=use_doc_unwarping,
                    )
                )
                doc_preprocessor_image = doc_preprocessor_res["output_img"]
                doc_preprocessor_res["img_id"] = img_id
            else:
                doc_preprocessor_res = {}
                doc_preprocessor_image = image_array

            ########## [TODO]RT-DETR 检测结果有重复
            layout_det_res = next(self.layout_det_model(doc_preprocessor_image))

            if input_params["use_common_ocr"] or input_params["use_table_recognition"]:
                overall_ocr_res = next(self.common_ocr_pipeline(doc_preprocessor_image))
                overall_ocr_res["img_id"] = img_id
                dt_boxes = convert_points_to_boxes(overall_ocr_res["dt_polys"])
                overall_ocr_res["dt_boxes"] = dt_boxes
            else:
                overall_ocr_res = {}

            text_paragraphs_ocr_res = {}
            if input_params["use_common_ocr"]:
                text_paragraphs_ocr_res = self.get_text_paragraphs_ocr_res(
                    overall_ocr_res, layout_det_res
                )
                text_paragraphs_ocr_res["img_id"] = img_id

            table_res_list = []
            if input_params["use_table_recognition"]:
                table_region_id = 1
                for box_info in layout_det_res["boxes"]:
                    if box_info["label"].lower() in ["table"]:
                        crop_img_info = self._crop_by_boxes(
                            doc_preprocessor_image, [box_info]
                        )
                        crop_img_info = crop_img_info[0]
                        table_structure_pred = next(
                            self.table_structure_model(crop_img_info["img"])
                        )
                        table_recognition_res = get_table_recognition_res(
                            crop_img_info, table_structure_pred, overall_ocr_res
                        )
                        table_recognition_res["table_region_id"] = table_region_id
                        table_region_id += 1
                        table_res_list.append(table_recognition_res)

            seal_res_list = []
            if input_params["use_seal_recognition"]:
                seal_region_id = 1
                for box_info in layout_det_res["boxes"]:
                    if box_info["label"].lower() in ["seal"]:
                        crop_img_info = self._crop_by_boxes(
                            doc_preprocessor_image, [box_info]
                        )
                        crop_img_info = crop_img_info[0]
                        seal_ocr_res = next(
                            self.seal_ocr_pipeline(crop_img_info["img"])
                        )
                        seal_ocr_res["seal_region_id"] = seal_region_id
                        seal_region_id += 1
                        seal_res_list.append(seal_ocr_res)

            single_img_res = {
                "layout_det_res": layout_det_res,
                "doc_preprocessor_res": doc_preprocessor_res,
                "text_paragraphs_ocr_res": text_paragraphs_ocr_res,
                "table_res_list": table_res_list,
                "seal_res_list": seal_res_list,
                "input_params": input_params,
            }
            yield LayoutParsingResult(single_img_res)
