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
from typing import Tuple
import cv2
import PIL
import math
import random
import tempfile
import subprocess
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from ...common.result import BaseCVResult
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
)


class FormulaRecognitionResult(dict):
    """Layout Parsing Result"""

    def __init__(self, data) -> None:
        """Initializes a new instance of the class with the specified data."""
        super().__init__(data)

    def save_to_img(self, save_path: str) -> None:
        """
        Saves an image with overlaid formula recognition results.

        This function attempts to save an image with recognized formulas highlighted
        and annotated. It verifies the environment setup before proceeding and logs
        a warning if the necessary rendering engine is not installed. The output image
        consists of two halves: the left side shows the original image with bounding
        boxes, and the right side shows the recognized formulas.

        Args:
            save_path (str): The directory path where the output image will be saved.

        Returns:
            None
        """
        try:
            env_valid()
        except subprocess.CalledProcessError as e:
            logging.warning(
                "Please refer to 2.3 Formula Recognition Pipeline Visualization in Formula Recognition Pipeline Tutorial to install the LaTeX rendering engine at first."
            )
            return None
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        img_id = self["img_id"]
        img_name = self["img_name"]
        if len(self["layout_det_res"]) <= 0:
            return
        image = Image.fromarray(self["layout_det_res"]["input_img"])
        h, w = image.height, image.width
        img_left = image.copy()
        img_right = np.ones((h, w, 3), dtype=np.uint8) * 255
        random.seed(0)
        draw_left = ImageDraw.Draw(img_left)

        formula_save_path = os.path.join(save_path, "formula_img_{}.jpg".format(img_id))
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
        img_show.save(formula_save_path)

    def save_results(self, save_path: str) -> None:
        """Save the formula recognition results to the specified directory.

        Args:
            save_path (str): The directory path to save the results.
        """
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.isdir(save_path):
            return

        img_id = self["img_id"]
        layout_det_res = self["layout_det_res"]
        if len(layout_det_res) > 0:
            save_img_path = Path(save_path) / f"layout_det_result_img{img_id}.jpg"
            layout_det_res.save_to_img(save_img_path)
        self.save_to_img(save_path)
        input_params = self["input_params"]
        if input_params["use_doc_preprocessor"]:
            save_img_path = Path(save_path) / f"doc_preprocessor_result_img{img_id}.jpg"
            self["doc_preprocessor_res"].save_to_img(save_img_path)
        for tno in range(len(self["formula_res_list"])):
            formula_res = self["formula_res_list"][tno]
            formula_region_id = formula_res["formula_region_id"]
            save_img_path = (
                Path(save_path)
                / f"formula_res_img{img_id}_region{formula_region_id}.jpg"
            )
            formula_res.save_to_img(save_img_path)
        return


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
