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
from pathlib import Path


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

        img_id = self["img_id"]
        layout_det_res = self["layout_det_res"]
        save_img_path = Path(save_path) / f"layout_det_result_img{img_id}.jpg"
        layout_det_res.save_to_img(save_img_path)

        input_params = self["input_params"]
        if input_params["use_doc_preprocessor"]:
            save_img_path = Path(save_path) / f"doc_preprocessor_result_img{img_id}.jpg"
            self["doc_preprocessor_res"].save_to_img(save_img_path)

        if input_params["use_general_ocr"]:
            save_img_path = (
                Path(save_path) / f"text_paragraphs_ocr_result_img{img_id}.jpg"
            )
            self["text_paragraphs_ocr_res"].save_to_img(save_img_path)

        if input_params["use_general_ocr"] or input_params["use_table_recognition"]:
            save_img_path = Path(save_path) / f"overall_ocr_result_img{img_id}.jpg"
            self["overall_ocr_res"].save_to_img(save_img_path)

        if input_params["use_table_recognition"]:
            for tno in range(len(self["table_res_list"])):
                table_res = self["table_res_list"][tno]
                table_region_id = table_res["table_region_id"]
                save_img_path = (
                    Path(save_path)
                    / f"table_res_cell_img{img_id}_region{table_region_id}.jpg"
                )
                table_res.save_to_img(save_img_path)
                save_html_path = (
                    Path(save_path)
                    / f"table_res_img{img_id}_region{table_region_id}.html"
                )
                table_res.save_to_html(save_html_path)
                save_xlsx_path = (
                    Path(save_path)
                    / f"table_res_img{img_id}_region{table_region_id}.xlsx"
                )
                table_res.save_to_xlsx(save_xlsx_path)

        if input_params["use_seal_recognition"]:
            for sno in range(len(self["seal_res_list"])):
                seal_res = self["seal_res_list"][sno]
                seal_region_id = seal_res["seal_region_id"]
                save_img_path = (
                    Path(save_path) / f"seal_res_img{img_id}_region{seal_region_id}.jpg"
                )
                seal_res.save_to_img(save_img_path)
        return
