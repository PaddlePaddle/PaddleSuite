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
from PIL import Image, ImageDraw, ImageFont

from ....utils.fonts import PINGFANG_FONT_FILE_PATH, create_font
from ..components import CVResult


class DocPreprocessorResult(CVResult):

    def save_to_img(self, save_path, *args, **kwargs):
        if not str(save_path).lower().endswith((".jpg", ".png")):
            img_id = self["img_id"]
            save_path = save_path + "/res_doc_preprocess_%d.jpg" % img_id
        super().save_to_img(save_path, *args, **kwargs)

    def _to_img(self):
        """draw doc preprocess result"""
        image = self["input_image"]
        angle = self["angle"]
        rot_img = self["rot_img"]
        output_img = self["output_img"]
        h, w = image.shape[0:2]
        img_show = Image.new("RGB", (w * 3, h + 25), (255, 255, 255))
        img_show.paste(Image.fromarray(image), (0, 0, w, h))
        img_show.paste(Image.fromarray(rot_img), (w, 0, w * 2, h))
        img_show.paste(Image.fromarray(output_img), (w * 2, 0, w * 3, h))

        draw_text = ImageDraw.Draw(img_show)
        txt_list = ["Original Image", "Rotated Image", "Unwarping Image"]
        for tno in range(len(txt_list)):
            txt = txt_list[tno]
            font = create_font(txt, (w, 20), PINGFANG_FONT_FILE_PATH)
            draw_text.text([10 + w * tno, h + 2], txt, fill=(0, 0, 0), font=font)
        return img_show
