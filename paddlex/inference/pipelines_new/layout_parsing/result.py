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

import math
import random
import numpy as np
import cv2
import PIL
import os
from PIL import Image, ImageDraw, ImageFont

from ....utils.fonts import PINGFANG_FONT_FILE_PATH
from ..components import CVResult, HtmlMixin, XlsxMixin

from typing import Any, Dict, Optional


class TableRecognitionResult(CVResult, HtmlMixin, XlsxMixin):
    """table recognition result"""

    def __init__(self, data: Dict) -> None:
        """Initializes the object with given data and sets up mixins for HTML and XLSX processing."""
        super().__init__(data)
        HtmlMixin.__init__(self)  # Initializes the HTML mixin functionality
        XlsxMixin.__init__(self)  # Initializes the XLSX mixin functionality

    def save_to_html(self, save_path: str, *args, **kwargs) -> None:
        """
        Save the content to an HTML file.

        Args:
            save_path (str): The path to save the HTML file. If the path does not end with '.html',
                          it will append '/res_table_%d.html' % self['table_region_id'] to the path.
            *args: Additional positional arguments to be passed to the superclass method.
            **kwargs: Additional keyword arguments to be passed to the superclass method.

        Returns:
            None
        """
        if not str(save_path).lower().endswith(".html"):
            save_path = save_path + "/res_table_%d.html" % self["table_region_id"]
        super().save_to_html(save_path, *args, **kwargs)

    def _to_html(self) -> str:
        """Converts the prediction to its corresponding HTML representation.

        Returns:
            str: The HTML string representation of the prediction.
        """
        return self["pred_html"]

    def save_to_xlsx(self, save_path: str, *args, **kwargs) -> None:
        """
        Save the content to an Excel file (.xlsx).

        If the save_path does not end with '.xlsx', it appends a default filename
        based on the table_region_id attribute.

        Args:
            save_path (str): The path where the Excel file should be saved.
            *args: Additional positional arguments passed to the superclass method.
            **kwargs: Additional keyword arguments passed to the superclass method.

        Returns:
            None
        """
        if not str(save_path).lower().endswith(".xlsx"):
            save_path = save_path + "/res_table_%d.xlsx" % self["table_region_id"]
        super().save_to_xlsx(save_path, *args, **kwargs)

    def _to_xlsx(self) -> str:
        """Converts the prediction HTML to an XLSX file path.

        Returns:
            str: The path to the XLSX file containing the prediction data.
        """
        return self["pred_html"]

    def save_to_img(self, save_path: str, *args, **kwargs) -> None:
        """
        Save the table and OCR result images to the specified path.

        Args:
            save_path (str): The directory path to save the images.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            None

        Raises:
            No specific exceptions are raised.

        Notes:
            - If save_path does not end with '.jpg' or '.png', the function appends '_res_table_cell_%d.jpg' and '_res_table_ocr_%d.jpg' to save_path
              with table_region_id respectively for table cell and OCR images.
            - The OCR result image is saved first with '_res_table_ocr_%d.jpg'.
            - Then the table image is saved with '_res_table_cell_%d.jpg'.
            - Calls the superclass's save_to_img method to save the table image.
        """
        if not str(save_path).lower().endswith((".jpg", ".png")):
            ocr_save_path = (
                save_path + "/res_table_ocr_%d.jpg" % self["table_region_id"]
            )
            save_path = save_path + "/res_table_cell_%d.jpg" % self["table_region_id"]
        self["table_ocr_pred"].save_to_img(ocr_save_path)
        super().save_to_img(save_path, *args, **kwargs)

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


class LayoutParsingResult(dict):
    """Layout Parsing Result"""

    def __init__(self, data) -> None:
        """Initializes a new instance of the class with the specified data."""
        super().__init__(data)

    def save_results(self, save_path: str) -> None:
        """Save the layout parsing results to the specified directory.

        Args:
            save_path (str): The directory path to save the results.
        """

        if not os.path.isdir(save_path):
            return

        layout_det_res = self["layout_det_res"]
        save_img_path = save_path + "/layout_det_result.jpg"
        layout_det_res.save_to_img(save_img_path)

        input_params = self["input_params"]
        if input_params["use_doc_preprocessor"]:
            save_img_path = save_path + "/doc_preprocessor_result.jpg"
            self["doc_preprocessor_res"].save_to_img(save_img_path)

        if input_params["use_general_ocr"]:
            save_img_path = save_path + "/text_paragraphs_ocr_result.jpg"
            self["text_paragraphs_ocr_res"].save_to_img(save_img_path)

        if input_params["use_table_recognition"]:
            for tno in range(len(self["table_res_list"])):
                table_res = self["table_res_list"][tno]
                table_res.save_to_img(save_path)
                table_res.save_to_html(save_path)
                table_res.save_to_xlsx(save_path)

        if input_params["use_seal_recognition"]:
            for sno in range(len(self["seal_res_list"])):
                seal_res = self["seal_res_list"][sno]
                save_img_path = (
                    save_path
                    + "/seal_%d_recognition_result.jpg" % seal_res["seal_region_id"]
                )
                seal_res.save_to_img(save_img_path)
        return
