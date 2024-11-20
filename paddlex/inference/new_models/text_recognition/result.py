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

import PIL
from PIL import Image, ImageDraw, ImageFont

from ....utils.fonts import PINGFANG_FONT_FILE_PATH
from ..base import CVResult


class TextRecResult(CVResult):
    INPUT_KEYS = ["input_path", "text", "score"]

    def _to_img(self):
        """Draw label on image"""
        image = self._img_reader.read(self["input_path"])
        rec_text = self["text"]
        rec_score = self["score"]
        image = image.convert("RGB")
        image_width, image_height = image.size
        text = f"{rec_text} ({rec_score})"
        font = self.adjust_font_size(image_width, text, PINGFANG_FONT_FILE_PATH)
        row_height = font.getbbox(text)[3]
        new_image_height = image_height + int(row_height * 1.2)
        new_image = Image.new("RGB", (image_width, new_image_height), (255, 255, 255))
        new_image.paste(image, (0, 0))

        draw = ImageDraw.Draw(new_image)
        draw.text(
            (0, image_height),
            text,
            fill=(0, 0, 0),
            font=font,
        )
        return new_image

    def adjust_font_size(self, image_width, text, font_path):
        font_size = int(image_width * 0.06)
        font = ImageFont.truetype(font_path, font_size)

        if int(PIL.__version__.split(".")[0]) < 10:
            text_width, _ = font.getsize(text)
        else:
            text_width, _ = font.getbbox(text)[2:]

        while text_width > image_width:
            font_size -= 1
            font = ImageFont.truetype(font_path, font_size)
            if int(PIL.__version__.split(".")[0]) < 10:
                text_width, _ = font.getsize(text)
            else:
                text_width, _ = font.getbbox(text)[2:]

        return font
