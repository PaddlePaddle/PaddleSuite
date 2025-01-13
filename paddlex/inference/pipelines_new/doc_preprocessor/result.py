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
import math
import random
from pathlib import Path
import numpy as np
import cv2
import PIL
from PIL import Image, ImageDraw, ImageFont
from ....utils.fonts import PINGFANG_FONT_FILE_PATH, create_font
from ...common.result import BaseCVResult


class DocPreprocessorResult(BaseCVResult):
    """doc preprocessor result"""

    def save_to_img(self, save_path: str, *args, **kwargs) -> None:
        """
        Save the image to the specified path.

        Args:
            save_path (str): The path to save the image.
                If the path does not end with '.jpg' or '.png', it appends '_res_doc_preprocess_<img_id>.jpg'
                to the path where <img_id> is retrieved from the object's 'img_id' attribute.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            None
        """
        if not str(save_path).lower().endswith((".jpg", ".png")):
            img_id = self["img_id"]
            save_path = Path(save_path) / f"res_doc_preprocess_{img_id}.jpg"
        super().save_to_img(save_path, *args, **kwargs)

    def _to_img(self) -> Dict[str, Image.Image]:
        """
        Generate an image combining the original, rotated, and unwarping images.

        Returns:
            Dict[Image.Image]: A new image that displays the rotated, and unwarping images.
        """
        imgs = {"preprocessed_img": Image.fromarray(self["output_img"][:, :, ::-1])}
        if self["rot_img"] is not None:
            imgs["rotated_img"] = Image.fromarray(self["rot_img"][:, :, ::-1])
        return imgs
