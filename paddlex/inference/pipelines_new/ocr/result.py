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

from pathlib import Path
import math
import random
import numpy as np
import cv2
import PIL
from PIL import Image, ImageDraw, ImageFont
from ....utils.fonts import PINGFANG_FONT_FILE_PATH, create_font
from ...common.result import BaseCVResult


class OCRResult(BaseCVResult):
    """OCR result"""

    def save_to_img(self, save_path: str, *args, **kwargs) -> None:
        """
        Save the image to the specified path with the appropriate extension.

        If the save_path does not end with '.jpg' or '.png', it appends '_res_ocr_<img_id>.jpg'
        to the path where <img_id> is the id of the image.

        Args:
            save_path (str): The path to save the image.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        input_params = self["input_params"]
        img_id = self["img_id"]
        if input_params["use_doc_preprocessor"]:
            save_img_path = Path(save_path) / f"doc_preprocessor_result_img_{img_id}.jpg"
            self["doc_preprocessor_res"].save_to_img(save_img_path)

        if not str(save_path).lower().endswith((".jpg", ".png")):
            save_path = Path(save_path) / f"res_ocr_{img_id}.jpg"

        super().save_to_img(save_path, *args, **kwargs)

    def get_minarea_rect(self, points: np.ndarray) -> np.ndarray:
        """
        Get the minimum area rectangle for the given points using OpenCV.

        Args:
            points (np.ndarray): An array of 2D points.

        Returns:
            np.ndarray: An array of 2D points representing the corners of the minimum area rectangle
                     in a specific order (clockwise or counterclockwise starting from the top-left corner).
        """
        bounding_box = cv2.minAreaRect(points)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_a, index_b, index_c, index_d = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_a = 0
            index_d = 1
        else:
            index_a = 1
            index_d = 0
        if points[3][1] > points[2][1]:
            index_b = 2
            index_c = 3
        else:
            index_b = 3
            index_c = 2

        box = np.array(
            [points[index_a], points[index_b], points[index_c], points[index_d]]
        ).astype(np.int32)

        return box

    def _to_img(self) -> PIL.Image:
        """
        Converts the internal data to a PIL Image with detection and recognition results.

        Returns:
            PIL.Image: An image with detection boxes, texts, and scores blended on it.
        """

        # TODO(gaotingquan): mv to postprocess
        drop_score = 0.5

        boxes = self["dt_polys"]
        txts = self["rec_text"]
        scores = self["rec_score"]
        image = self["doc_preprocessor_image"]
        h, w = image.shape[0:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_left = Image.fromarray(image_rgb)
        img_right = np.ones((h, w, 3), dtype=np.uint8) * 255
        random.seed(0)
        draw_left = ImageDraw.Draw(img_left)
        if txts is None or len(txts) != len(boxes):
            txts = [None] * len(boxes)
        for idx, (box, txt) in enumerate(zip(boxes, txts)):
            try:
                if scores is not None and scores[idx] < drop_score:
                    continue
                color = (
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255),
                )
                box = np.array(box)
                if len(box) > 4:
                    pts = [(x, y) for x, y in box.tolist()]
                    draw_left.polygon(pts, outline=color, width=8)
                    box = self.get_minarea_rect(box)
                    height = int(0.5 * (max(box[:, 1]) - min(box[:, 1])))
                    box[:2, 1] = np.mean(box[:, 1])
                    box[2:, 1] = np.mean(box[:, 1]) + min(20, height)
                draw_left.polygon(box, fill=color)
                img_right_text = draw_box_txt_fine(
                    (w, h), box, txt, PINGFANG_FONT_FILE_PATH
                )
                pts = np.array(box, np.int32).reshape((-1, 1, 2))
                cv2.polylines(img_right_text, [pts], True, color, 1)
                img_right = cv2.bitwise_and(img_right, img_right_text)
            except:
                continue

        img_left = Image.blend(Image.fromarray(image_rgb), img_left, 0.5)
        img_show = Image.new("RGB", (w * 2, h), (255, 255, 255))
        img_show.paste(img_left, (0, 0, w, h))
        img_show.paste(Image.fromarray(img_right), (w, 0, w * 2, h))
        return img_show


# Adds a function comment according to Google Style Guide
def draw_box_txt_fine(
    img_size: tuple, box: np.ndarray, txt: str, font_path: str
) -> np.ndarray:
    """
    Draws text in a box on an image with fine control over size and orientation.

    Args:
        img_size (tuple): The size of the output image (width, height).
        box (np.ndarray): A 4x2 numpy array defining the corners of the box in (x, y) order.
        txt (str): The text to draw inside the box.
        font_path (str): The path to the font file to use for drawing the text.

    Returns:
        np.ndarray: An image with the text drawn in the specified box.
    """
    box_height = int(
        math.sqrt((box[0][0] - box[3][0]) ** 2 + (box[0][1] - box[3][1]) ** 2)
    )
    box_width = int(
        math.sqrt((box[0][0] - box[1][0]) ** 2 + (box[0][1] - box[1][1]) ** 2)
    )

    if box_height > 2 * box_width and box_height > 30:
        img_text = Image.new("RGB", (box_height, box_width), (255, 255, 255))
        draw_text = ImageDraw.Draw(img_text)
        if txt:
            font = create_font(txt, (box_height, box_width), font_path)
            draw_text.text([0, 0], txt, fill=(0, 0, 0), font=font)
        img_text = img_text.transpose(Image.ROTATE_270)
    else:
        img_text = Image.new("RGB", (box_width, box_height), (255, 255, 255))
        draw_text = ImageDraw.Draw(img_text)
        if txt:
            font = create_font(txt, (box_width, box_height), font_path)
            draw_text.text([0, 0], txt, fill=(0, 0, 0), font=font)

    pts1 = np.float32(
        [[0, 0], [box_width, 0], [box_width, box_height], [0, box_height]]
    )
    pts2 = np.array(box, dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts1, pts2)

    img_text = np.array(img_text, dtype=np.uint8)
    img_right_text = cv2.warpPerspective(
        img_text,
        M,
        img_size,
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return img_right_text
