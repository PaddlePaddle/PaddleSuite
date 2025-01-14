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
from typing import Tuple, List, Dict, Any
import cv2
import PIL
import math
import copy
import random
import tempfile
import subprocess
import numpy as np
from pathlib import Path
import PIL
from PIL import Image, ImageDraw, ImageFont

from ...common.result import BaseCVResult
from ...common.result.mixin import JsonMixin, ImgMixin, StrMixin
from ....utils import logging
from ....utils.fonts import PINGFANG_FONT_FILE_PATH
from ...models_new.formula_recognition.result import (
    get_align_equation,
    generate_tex_file,
    generate_pdf_file,
    env_valid,
    pdf2img,
    create_font,
    crop_white_area,
    draw_box_txt_fine,
    draw_formula_module,
)


class FormulaRecognitionResult(BaseCVResult):
    """Formula Recognition Result"""

    def _to_str(self, *args: List, **kwargs: Dict) -> Dict[str, Any]:
        """
        Convert the object to a string representation.

        Args:
            *args: Additional positional arguments for the superclass's _to_str method.
            **kwargs: Additional keyword arguments for the superclass's _to_str method.

        Returns:
            str: A string representation of the object with specified fields removed.
        """
        data = copy.deepcopy(self)
        for tno in range(len(self["formula_res_list"])):
            data["formula_res_list"][tno].pop("input_path", None)
            data["formula_res_list"][tno].pop("input_img", None)
        data.pop("layout_det_res", None)
        data.pop("doc_preprocessor_res", None)
        return StrMixin._to_str(data, *args, **kwargs)

    def _to_json(self, *args: List, **kwargs: Dict) -> Dict[str, Any]:
        """
        Convert the object to a JSON-compatible dictionary.

        Args:
            *args: Additional positional arguments for the superclass's _to_json method.
            **kwargs: Additional keyword arguments for the superclass's _to_json method.

        Returns:
            Dict[str, Any]: A JSON-compatible dictionary representation of the object
            with specified fields removed.
        """
        data = copy.deepcopy(self)
        for tno in range(len(self["formula_res_list"])):
            data["formula_res_list"][tno].pop("input_path", None)
            data["formula_res_list"][tno].pop("input_img", None)
        data["layout_det_res"].pop("input_path", None)
        data["layout_det_res"].pop("input_img", None)
        data["doc_preprocessor_res"].pop("output_img", None)
        return JsonMixin._to_json(data, *args, **kwargs)

    def _to_img(self) -> PIL.Image:
        """
        Converts the internal data to a PIL Image with detection and recognition results.

        Returns:
            PIL.Image: An image with detection boxes, texts, and scores blended on it.
        """
        image = Image.fromarray(self["doc_preprocessor_res"]["output_img"])
        try:
            env_valid()
        except subprocess.CalledProcessError as e:
            logging.warning(
                "Please refer to 2.3 Formula Recognition Pipeline Visualization in Formula Recognition Pipeline Tutorial to install the LaTeX rendering engine at first."
            )
            return image

        if len(self["layout_det_res"]) <= 0:
            image = np.array(image.convert("RGB"))
            rec_formula = self["formula_res_list"][0]["rec_formula"]
            xywh = crop_white_area(image)
            if xywh is not None:
                x, y, w, h = xywh
                image = image[y : y + h, x : x + w]
            image = Image.fromarray(image)
            image_width, image_height = image.size
            box = [
                [0, 0],
                [image_width, 0],
                [image_width, image_height],
                [0, image_height],
            ]
            try:
                img_formula = draw_formula_module(
                    image.size, box, rec_formula, is_debug=False
                )
                img_formula = Image.fromarray(img_formula)
                render_width, render_height = img_formula.size
                resize_height = render_height
                resize_width = int(resize_height * image_width / image_height)
                image = image.resize((resize_width, resize_height), Image.LANCZOS)

                new_image_width = image.width + int(render_width) + 10
                new_image = Image.new(
                    "RGB", (new_image_width, render_height), (255, 255, 255)
                )
                new_image.paste(image, (0, 0))
                new_image.paste(img_formula, (image.width + 10, 0))
                return {f"formula_res_img": new_image}
            except subprocess.CalledProcessError as e:
                logging.warning("Syntax error detected in formula, rendering failed.")
                return {f"formula_res_img": image}

        h, w = image.height, image.width
        img_left = image.copy()
        img_right = np.ones((h, w, 3), dtype=np.uint8) * 255
        random.seed(0)
        draw_left = ImageDraw.Draw(img_left)

        formula_res_list = self["formula_res_list"]
        for tno in range(len(self["formula_res_list"])):
            formula_res = self["formula_res_list"][tno]
            formula_region_id = formula_res["formula_region_id"]
            formula = str(formula_res["rec_formula"])
            dt_polys = formula_res["dt_polys"]
            x1, y1, x2, y2 = list(dt_polys)
            try:
                color = (
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255),
                )
                box = [x1, y1, x2, y1, x2, y2, x1, y2]
                box = np.array(box).reshape([-1, 2])
                pts = [(x, y) for x, y in box.tolist()]
                draw_left.polygon(pts, outline=color, width=8)
                draw_left.polygon(box, fill=color)
                img_right_text = draw_box_formula_fine(
                    (w, h),
                    box,
                    formula,
                    is_debug=False,
                )
                pts = np.array(box, np.int32).reshape((-1, 1, 2))
                cv2.polylines(img_right_text, [pts], True, color, 1)
                img_right = cv2.bitwise_and(img_right, img_right_text)
            except subprocess.CalledProcessError as e:
                logging.warning("Syntax error detected in formula, rendering failed.")
                continue
        img_left = Image.blend(image, img_left, 0.5)
        img_show = Image.new("RGB", (int(w * 2), h), (255, 255, 255))
        img_show.paste(img_left, (0, 0, w, h))
        img_show.paste(Image.fromarray(img_right), (w, 0, w * 2, h))
        model_settings = self["model_settings"]
        if model_settings["use_doc_preprocessor"]:
            return {
                **self["doc_preprocessor_res"].img,
                f"formula_res_img": img_show,
            }
        else:
            return {f"formula_res_img": img_show}


def draw_box_formula_fine(
    img_size: Tuple[int, int], box: np.ndarray, formula: str, is_debug: bool = False
) -> np.ndarray:
    """draw box formula for pipeline"""
    """
    Draw box formula for pipeline.

    This function generates a LaTeX formula image and transforms it to fit
    within a specified bounding box on a larger image. If the rendering fails,
    it will write "Rendering Failed" inside the box.

    Args:
        img_size (Tuple[int, int]): The size of the image (width, height).
        box (np.ndarray): A numpy array representing the four corners of the bounding box.
        formula (str): The LaTeX formula to render.
        is_debug (bool, optional): If True, enables debug mode. Defaults to False.

    Returns:
        np.ndarray: An image array with the rendered formula inside the specified box.
    """
    box_height = int(
        math.sqrt((box[0][0] - box[3][0]) ** 2 + (box[0][1] - box[3][1]) ** 2)
    )
    box_width = int(
        math.sqrt((box[0][0] - box[1][0]) ** 2 + (box[0][1] - box[1][1]) ** 2)
    )
    with tempfile.TemporaryDirectory() as td:
        tex_file_path = os.path.join(td, "temp.tex")
        pdf_file_path = os.path.join(td, "temp.pdf")
        img_file_path = os.path.join(td, "temp.jpg")
        generate_tex_file(tex_file_path, formula)
        if os.path.exists(tex_file_path):
            generate_pdf_file(tex_file_path, td, is_debug)
        formula_img = None
        if os.path.exists(pdf_file_path):
            formula_img = pdf2img(pdf_file_path, img_file_path, is_padding=False)
        if formula_img is not None:
            formula_h, formula_w = formula_img.shape[:-1]
            resize_height = box_height
            resize_width = formula_w * resize_height / formula_h
            formula_img = cv2.resize(
                formula_img, (int(resize_width), int(resize_height))
            )
            formula_h, formula_w = formula_img.shape[:-1]
            pts1 = np.float32(
                [[0, 0], [box_width, 0], [box_width, box_height], [0, box_height]]
            )
            pts2 = np.array(box, dtype=np.float32)
            M = cv2.getPerspectiveTransform(pts1, pts2)
            formula_img = np.array(formula_img, dtype=np.uint8)
            img_right_text = cv2.warpPerspective(
                formula_img,
                M,
                img_size,
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255, 255, 255),
            )
        else:
            img_right_text = draw_box_txt_fine(
                img_size, box, "Rendering Failed", PINGFANG_FONT_FILE_PATH
            )
        return img_right_text
