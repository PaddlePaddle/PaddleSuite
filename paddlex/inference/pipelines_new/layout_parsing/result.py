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

from typing import Dict
import numpy as np
import copy
import cv2
from pathlib import Path
from PIL import Image, ImageDraw
from .utils import recursive_img_array2path,get_layout_ordering
from ...common.result import BaseCVResult, HtmlMixin, XlsxMixin, StrMixin, JsonMixin, MarkdownMixin


class LayoutParsingResult(BaseCVResult, HtmlMixin, XlsxMixin, MarkdownMixin):
    """Layout Parsing Result"""

    def __init__(self, data) -> None:
        """Initializes a new instance of the class with the specified data."""
        super().__init__(data)
        HtmlMixin.__init__(self)
        XlsxMixin.__init__(self)
        MarkdownMixin.__init__(self)
        JsonMixin.__init__(self)
        self.already_sorted = False

    def _to_img(self) -> Dict[str, np.ndarray]:
        res_img_dict = {}
        model_settings = self["model_settings"]
        if model_settings["use_doc_preprocessor"]:
            res_img_dict.update(**self["doc_preprocessor_res"].img)
        res_img_dict["layout_det_res"] = self["layout_det_res"].img["res"]

        if model_settings["use_general_ocr"] or model_settings["use_table_recognition"]:
            res_img_dict["overall_ocr_res"] = self["overall_ocr_res"].img["ocr_res_img"]

        if model_settings["use_general_ocr"]:
            general_ocr_res = copy.deepcopy(self["overall_ocr_res"])
            general_ocr_res["rec_polys"] = self["text_paragraphs_ocr_res"]["rec_polys"]
            general_ocr_res["rec_texts"] = self["text_paragraphs_ocr_res"]["rec_texts"]
            general_ocr_res["rec_scores"] = self["text_paragraphs_ocr_res"][
                "rec_scores"
            ]
            general_ocr_res["rec_boxes"] = self["text_paragraphs_ocr_res"]["rec_boxes"]
            res_img_dict["text_paragraphs_ocr_res"] = general_ocr_res.img["ocr_res_img"]

        if model_settings["use_table_recognition"] and len(self["table_res_list"]) > 0:
            table_cell_img = copy.deepcopy(self["doc_preprocessor_res"]["output_img"])
            for sno in range(len(self["table_res_list"])):
                table_res = self["table_res_list"][sno]
                cell_box_list = table_res["cell_box_list"]
                for box in cell_box_list:
                    x1, y1, x2, y2 = [int(pos) for pos in box]
                    cv2.rectangle(table_cell_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            res_img_dict["table_cell_img"] = table_cell_img

        if model_settings["use_seal_recognition"] and len(self["seal_res_list"]) > 0:
            for sno in range(len(self["seal_res_list"])):
                seal_res = self["seal_res_list"][sno]
                seal_region_id = seal_res["seal_region_id"]
                sub_seal_res_dict = seal_res.img
                key = f"seal_res_region{seal_region_id}"
                res_img_dict[key] = sub_seal_res_dict["ocr_res_img"]

        if (
            model_settings["use_formula_recognition"]
            and len(self["formula_res_list"]) > 0
        ):
            for sno in range(len(self["formula_res_list"])):
                formula_res = self["formula_res_list"][sno]
                formula_region_id = formula_res["formula_region_id"]
                sub_formula_res_dict = formula_res.img
                key = f"formula_res_region{formula_region_id}"
                res_img_dict[key] = sub_formula_res_dict

        return res_img_dict

    def _to_str(self, *args, **kwargs) -> Dict[str, str]:
        """Converts the instance's attributes to a dictionary and then to a string.

        Args:
            *args: Additional positional arguments passed to the base class method.
            **kwargs: Additional keyword arguments passed to the base class method.

        Returns:
            Dict[str, str]: A dictionary with the instance's attributes converted to strings.
        """
        data = {}
        data["input_path"] = self["input_path"]
        model_settings = self["model_settings"]
        data["model_settings"] = model_settings
        if self["model_settings"]["use_doc_preprocessor"]:
            data["doc_preprocessor_res"] = self["doc_preprocessor_res"].str["res"]
        data["layout_det_res"] = self["layout_det_res"].str["res"]
        if model_settings["use_general_ocr"] or model_settings["use_table_recognition"]:
            data["overall_ocr_res"] = self["overall_ocr_res"].str["res"]
        if model_settings["use_general_ocr"]:
            general_ocr_res = {}
            general_ocr_res["rec_polys"] = self["text_paragraphs_ocr_res"]["rec_polys"]
            general_ocr_res["rec_texts"] = self["text_paragraphs_ocr_res"]["rec_texts"]
            general_ocr_res["rec_scores"] = self["text_paragraphs_ocr_res"][
                "rec_scores"
            ]
            general_ocr_res["rec_boxes"] = self["text_paragraphs_ocr_res"]["rec_boxes"]
            data["text_paragraphs_ocr_res"] = general_ocr_res
        if model_settings["use_table_recognition"] and len(self["table_res_list"]) > 0:
            data["table_res_list"] = []
            for sno in range(len(self["table_res_list"])):
                table_res = self["table_res_list"][sno]
                data["table_res_list"].append(table_res.str["res"])
        if model_settings["use_seal_recognition"] and len(self["seal_res_list"]) > 0:
            data["seal_res_list"] = []
            for sno in range(len(self["seal_res_list"])):
                seal_res = self["seal_res_list"][sno]
                data["seal_res_list"].append(seal_res.str["res"])
        if (
            model_settings["use_formula_recognition"]
            and len(self["formula_res_list"]) > 0
        ):
            data["formula_res_list"] = []
            for sno in range(len(self["formula_res_list"])):
                formula_res = self["formula_res_list"][sno]
                data["formula_res_list"].append(formula_res.str["res"])

        return StrMixin._to_str(data, *args, **kwargs)

    def _to_json(self, *args, **kwargs) -> Dict[str, str]:
        """
        Converts the object's data to a JSON dictionary.

        Args:
            *args: Positional arguments passed to the JsonMixin._to_json method.
            **kwargs: Keyword arguments passed to the JsonMixin._to_json method.

        Returns:
            Dict[str, str]: A dictionary containing the object's data in JSON format.
        """
        data = {}
        data["input_path"] = self["input_path"]
        model_settings = self["model_settings"]
        data["model_settings"] = model_settings
        if self["model_settings"]["use_doc_preprocessor"]:
            data["doc_preprocessor_res"] = self["doc_preprocessor_res"].json["res"]
        data["layout_det_res"] = self["layout_det_res"].json["res"]
        if model_settings["use_general_ocr"] or model_settings["use_table_recognition"]:
            data["overall_ocr_res"] = self["overall_ocr_res"].json["res"]
        if model_settings["use_general_ocr"]:
            general_ocr_res = {}
            general_ocr_res["rec_polys"] = self["text_paragraphs_ocr_res"]["rec_polys"]
            general_ocr_res["rec_texts"] = self["text_paragraphs_ocr_res"]["rec_texts"]
            general_ocr_res["rec_scores"] = self["text_paragraphs_ocr_res"][
                "rec_scores"
            ]
            general_ocr_res["rec_boxes"] = self["text_paragraphs_ocr_res"]["rec_boxes"]
            data["text_paragraphs_ocr_res"] = general_ocr_res
        if model_settings["use_table_recognition"] and len(self["table_res_list"]) > 0:
            data["table_res_list"] = []
            for sno in range(len(self["table_res_list"])):
                table_res = self["table_res_list"][sno]
                data["table_res_list"].append(table_res.json["res"])
        if model_settings["use_seal_recognition"] and len(self["seal_res_list"]) > 0:
            data["seal_res_list"] = []
            for sno in range(len(self["seal_res_list"])):
                seal_res = self["seal_res_list"][sno]
                data["seal_res_list"].append(seal_res.json["res"])
        if (
            model_settings["use_formula_recognition"]
            and len(self["formula_res_list"]) > 0
        ):
            data["formula_res_list"] = []
            for sno in range(len(self["formula_res_list"])):
                formula_res = self["formula_res_list"][sno]
                data["formula_res_list"].append(formula_res.json["res"])
        return JsonMixin._to_json(data, *args, **kwargs)

    def _to_html(self) -> Dict[str, str]:
        """Converts the prediction to its corresponding HTML representation.

        Returns:
            Dict[str, str]: The str type HTML representation result.
        """
        model_settings = self["model_settings"]
        res_html_dict = {}
        if model_settings["use_table_recognition"] and len(self["table_res_list"]) > 0:
            for sno in range(len(self["table_res_list"])):
                table_res = self["table_res_list"][sno]
                table_region_id = table_res["table_region_id"]
                key = f"table_{table_region_id}"
                res_html_dict[key] = table_res.html["pred"]
        return res_html_dict

    def _to_xlsx(self) -> Dict[str, str]:
        """Converts the prediction HTML to an XLSX file path.

        Returns:
            Dict[str, str]: The str type XLSX representation result.
        """
        model_settings = self["model_settings"]
        res_xlsx_dict = {}
        if model_settings["use_table_recognition"] and len(self["table_res_list"]) > 0:
            for sno in range(len(self["table_res_list"])):
                table_res = self["table_res_list"][sno]
                table_region_id = table_res["table_region_id"]
                key = f"table_{table_region_id}"
                res_xlsx_dict[key] = table_res.xlsx["pred"]
        return res_xlsx_dict

    def _to_pdf_order(self, save_path):
        """
        Save the layout ordering to an image file.

        Args:
            save_path (str or Path): The path where the image should be saved.
            font_path (str): Path to the font file used for drawing text.

        Returns:
            None
        """
        input_name = self["input_path"]
        if save_path.suffix.lower() not in (".jpg", ".png"):
            save_path = Path(save_path).with_suffix(f'{input_name}.jpg')
        else:
            save_path = save_path.with_suffix('')
        ordering_image_path = save_path.parent / f"{save_path.stem}_ordering.jpg"

        try:
            image = Image.fromarray(self["doc_preprocessor_res"]["output_img"])
        except IOError as e:
            print(f"Error opening image: {e}")
            return

        draw = ImageDraw.Draw(image,'RGBA')

        parsing_result = self['layout_parsing_result']
        for block_index, _ in enumerate(parsing_result):
            get_layout_ordering(
                block_index=block_index,
                no_mask_labels=['text', 'formula', 'algorithm', "reference", "content", "abstract"],
            )

            sub_blocks = parsing_result[block_index]['sub_blocks']
            for sub_block in sub_blocks:
                bbox = sub_block['layout_bbox']
                index = sub_block.get('index',None)
                label = sub_block['sub_label']
                fill_color = self.get_show_color(label)
                draw.rectangle(bbox, fill=fill_color)
                if index is not None:
                    text_position = (bbox[2]+2, bbox[1] - 10)
                    draw.text(text_position, str(index), fill="red")

        # Ensure the directory exists and save the image
        ordering_image_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving ordering image to {ordering_image_path}")
        image.save(str(ordering_image_path))

    def _to_markdown(self, save_path):
        """
        Save the parsing result to a Markdown file.

        Args:
            save_path (str or Path): The path where the Markdown file should be saved.

        Returns:
            None
        """
        save_path = Path(save_path)
        if not save_path.suffix.lower() == ".md":
            save_path = save_path / f"layout_parsing_result.md"

        parsing_result = self['layout_parsing_result']
        for block_index, _ in enumerate(parsing_result):
            get_layout_ordering(
                block_index=block_index,
                no_mask_labels=['text', 'formula', 'algorithm', 'reference', 'content', 'abstract'],
            )
        recursive_img_array2path(self['layout_parsing_result'], save_path.parent,labels=['img'])
        super()._to_markdown(save_path)        
    
    def get_show_color(self,label):
        label_colors = {
        'paragraph_title': (102, 102, 255, 100),  # Medium Blue (from 'titles_list')
        'doc_title': (255, 248, 220, 100),        # Cornsilk
        'table_title': (255, 255, 102, 100),      # Light Yellow (from 'tables_caption_list')
        'figure_title': (102, 178, 255, 100),     # Sky Blue (from 'imgs_caption_list')
        'chart_title': (221, 160, 221, 100),      # Plum
        'vision_footnote': (144, 238, 144, 100),  # Light Green
        'text': (153, 0, 76, 100),                # Deep Purple (from 'texts_list')
        'formula': (0, 255, 0, 100),              # Bright Green (from 'interequations_list')
        'abstract': (255, 239, 213, 100),         # Papaya Whip
        'content': (40, 169, 92, 100),            # Medium Green (from 'lists_list' and 'indexs_list')
        'seal': (158, 158, 158, 100),             # Neutral Gray (from 'dropped_bbox_list')
        'table': (204, 204, 0, 100),              # Olive Yellow (from 'tables_body_list')
        'image': (153, 255, 51, 100),             # Bright Green (from 'imgs_body_list')
        'figure': (153, 255, 51, 100),             # Bright Green (from 'imgs_body_list')
        'chart': (216, 191, 216, 100),            # Thistle
        'reference': (229, 255, 204, 100),        # Pale Yellow-Green (from 'tables_footnote_list')
        'algorithm': (255, 250, 240, 100)         # Floral White
        }
        default_color = (158, 158, 158, 100)
        return label_colors.get(label, default_color)