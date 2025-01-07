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
from typing import Dict
from pathlib import Path
import numpy as np
import cv2
from ...common.result import BaseCVResult, HtmlMixin, XlsxMixin


class SingleTableRecognitionResult(BaseCVResult, HtmlMixin, XlsxMixin):
    """table recognition result"""

    def __init__(self, data: Dict) -> None:
        """Initializes the object with given data and sets up mixins for HTML and XLSX processing."""
        super().__init__(data)
        HtmlMixin.__init__(self)  # Initializes the HTML mixin functionality
        XlsxMixin.__init__(self)  # Initializes the XLSX mixin functionality

    def _to_html(self) -> str:
        """Converts the prediction to its corresponding HTML representation.

        Returns:
            str: The HTML string representation of the prediction.
        """
        return self["pred_html"]

    def _to_xlsx(self) -> str:
        """Converts the prediction HTML to an XLSX file path.

        Returns:
            str: The path to the XLSX file containing the prediction data.
        """
        return self["pred_html"]

    def _to_img(self) -> np.ndarray:
        """
        Convert the input image with table OCR predictions to an image with cell boundaries highlighted.

        Returns:
            np.ndarray: The input image with cell boundaries highlighted in red.
        """
        input_img = self["table_ocr_pred"]["input_img"].copy()
        cell_box_list = self["cell_box_list"]
        for box in cell_box_list:
            x1, y1, x2, y2 = [int(pos) for pos in box]
            cv2.rectangle(input_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        return input_img


class TableRecognitionResult(dict):
    """Layout Parsing Result"""

    def __init__(self, data) -> None:
        """Initializes a new instance of the class with the specified data."""
        super().__init__(data)

    def save_results(self, save_path: str) -> None:
        """Save the table recognition results to the specified directory.

        Args:
            save_path (str): The directory path to save the results.
        """

        if not os.path.isdir(save_path):
            return

        img_id = self["img_id"]
        layout_det_res = self["layout_det_res"]
        if len(layout_det_res) > 0:
            save_img_path = Path(save_path) / f"layout_det_result_img{img_id}.jpg"
            layout_det_res.save_to_img(save_img_path)

        input_params = self["input_params"]
        if input_params["use_doc_preprocessor"]:
            save_img_path = Path(save_path) / f"doc_preprocessor_result_img{img_id}.jpg"
            self["doc_preprocessor_res"].save_to_img(save_img_path)

        save_img_path = Path(save_path) / f"overall_ocr_result_img{img_id}.jpg"
        self["overall_ocr_res"].save_to_img(save_img_path)

        for tno in range(len(self["table_res_list"])):
            table_res = self["table_res_list"][tno]
            table_region_id = table_res["table_region_id"]
            save_img_path = (
                Path(save_path)
                / f"table_res_cell_img{img_id}_region{table_region_id}.jpg"
            )
            table_res.save_to_img(save_img_path)
            save_html_path = (
                Path(save_path) / f"table_res_img{img_id}_region{table_region_id}.html"
            )
            table_res.save_to_html(save_html_path)
            save_xlsx_path = (
                Path(save_path) / f"table_res_img{img_id}_region{table_region_id}.xlsx"
            )
            table_res.save_to_xlsx(save_xlsx_path)

        return
