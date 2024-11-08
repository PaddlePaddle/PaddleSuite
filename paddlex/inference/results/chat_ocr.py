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

import copy
from pathlib import Path
from .base import BaseResult
from .utils.mixin import Base64Mixin


class LayoutParsingResult(BaseResult):
    """LayoutParsingResult"""

    pass


class VisualInfoResult(BaseResult):
    """VisualInfoResult"""

    pass


class VisualResult(BaseResult):
    """VisualInfoResult"""

    def __init__(self, data, page_id=None, src_input_name=None):
        super().__init__(data)
        self.page_id = page_id
        self.src_input_name = src_input_name

    def _to_str(self, _, *args, **kwargs):
        return super()._to_str(
            {"layout_parsing_result": self["layout_parsing_result"]}, *args, **kwargs
        )

    def get_target_name(self, save_path):
        if self.src_input_name.endswith(".pdf"):
            save_path = (
                Path(save_path)
                / f"{Path(self.src_input_name).stem}_pdf"
                / Path("page_{:04d}".format(self.page_id + 1))
            )
        else:
            save_path = Path(save_path) / f"{Path(self.src_input_name).stem}"
        return save_path

    def save_to_json(self, save_path):
        if not save_path.lower().endswith(("json")):
            save_path = self.get_target_name(save_path)
        else:
            save_path = Path(save_path).stem
        layout_save_path = f"{save_path}_layout.jpg"
        ocr_save_path = f"{save_path}_ocr.jpg"
        table_save_path = f"{save_path}_table"
        self["input_path"] = layout_save_path
        self["layout_result"]["input_path"] = layout_save_path
        self["layout_parsing_result"]["input_path"] = layout_save_path
        self["ocr_result"]["input_path"] = ocr_save_path
        table_result_num = len(self["table_result"])
        for idx in range(table_result_num):
            self["table_result"][idx]["input_path"] = "{}_{:04d}".format(
                table_save_path, idx + 1
            )

        if not str(save_path).endswith(".json"):
            save_path = "{}.json".format(save_path)
        super().save_to_json(save_path)

    def save_to_html(self, save_path):
        if not save_path.lower().endswith(("html")):
            save_path = self.get_target_name(save_path)
        else:
            save_path = Path(save_path).stem
        table_save_path = f"{save_path}_table"
        for idx, table_result in enumerate(self["table_result"]):
            basename = (Path(table_result["input_path"]).name).split(".")[0]
            table_result["input_path"] = Path(
                str(table_result["input_path"]).replace(
                    basename, "{}_{:04d}".format(table_save_path, idx + 1)
                )
            )
            table_result.save_to_html(save_path)

    def save_to_xlsx(self, save_path):
        if not save_path.lower().endswith(("xlsx")):
            save_path = self.get_target_name(save_path)
        else:
            save_path = Path(save_path).stem
        table_save_path = f"{save_path}_table"
        for idx, table_result in enumerate(self["table_result"]):
            basename = (Path(table_result["input_path"]).name).split(".")[0]
            table_result["input_path"] = Path(
                str(table_result["input_path"]).replace(
                    basename, "{}_{:04d}".format(table_save_path, idx + 1)
                )
            )
            table_result.save_to_xlsx(save_path)

    def save_to_img(self, save_path):
        if not save_path.lower().endswith((".jpg", ".png")):
            save_path = self.get_target_name(save_path)
        else:
            save_path = Path(save_path).stem

        oricls_save_path = f"{save_path}_oricls.jpg"
        oricls_result = self["oricls_result"]
        if oricls_result:
            oricls_result.save_to_img(oricls_save_path)
        uvdoc_save_path = f"{save_path}_uvdoc.jpg"
        unwarp_result = self["unwarp_result"]
        if unwarp_result:
            unwarp_result.save_to_img(uvdoc_save_path)
        curve_save_path = f"{save_path}_curve"
        curve_results = self["curve_result"]
        # TODO(): support list of result
        if isinstance(curve_results, dict):
            curve_results = [curve_results]
        for idx, curve_result in enumerate(curve_results):
            curve_result.save_to_img(f"{curve_save_path}_{idx}.jpg")
        layout_save_path = f"{save_path}_layout.jpg"
        layout_result = self["layout_result"]
        if layout_result:
            layout_result.save_to_img(layout_save_path)
        ocr_save_path = f"{save_path}_ocr.jpg"
        table_save_path = f"{save_path}_table"
        ocr_result = self["ocr_result"]
        if ocr_result:
            ocr_result.save_to_img(ocr_save_path)
        for idx, table_result in enumerate(self["table_result"]):
            table_result.save_to_img(f"{table_save_path}_{idx}.jpg")


class VectorResult(BaseResult, Base64Mixin):
    """VisualInfoResult"""

    def _to_base64(self):
        return self["vector"]


class RetrievalResult(BaseResult):
    """VisualInfoResult"""

    pass


class ChatResult(BaseResult):
    """VisualInfoResult"""

    pass
