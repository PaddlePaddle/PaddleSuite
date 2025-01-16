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

from abc import abstractmethod
import json
from pathlib import Path
import numpy as np
from PIL import Image
import pandas as pd
import re

from .....utils import logging
from ....utils.io import (
    JsonWriter,
    ImageReader,
    ImageWriter,
    CSVWriter,
    HtmlWriter,
    XlsxWriter,
    TextWriter,
    MarkdownWriter
)

#### [TODO] need tingquan to add explanatory notes


def _save_list_data(save_func, save_path, data, *args, **kwargs):
    save_path = Path(save_path)
    if data is None:
        return
    if isinstance(data, list):
        for idx, single in enumerate(data):
            save_func(
                (
                    save_path.parent / f"{save_path.stem}_{idx}{save_path.suffix}"
                ).as_posix(),
                single,
                *args,
                **kwargs,
            )
    save_func(save_path.as_posix(), data, *args, **kwargs)
    logging.info(f"The result has been saved in {save_path}.")


class StrMixin:
    @property
    def str(self):
        return self._to_str()

    def _to_str(self, data, json_format=False, indent=4, ensure_ascii=False):
        if json_format:
            return json.dumps(data.json, indent=indent, ensure_ascii=ensure_ascii)
        else:
            return str(data)

    def print(self, json_format=False, indent=4, ensure_ascii=False):
        str_ = self._to_str(
            self, json_format=json_format, indent=indent, ensure_ascii=ensure_ascii
        )
        logging.info(str_)


class JsonMixin:
    def __init__(self):
        self._json_writer = JsonWriter()
        self._show_funcs.append(self.save_to_json)

    def _to_json(self):
        def _format_data(obj):
            if isinstance(obj, np.float32):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return [_format_data(item) for item in obj.tolist()]
            elif isinstance(obj, pd.DataFrame):
                return obj.to_json(orient="records", force_ascii=False)
            elif isinstance(obj, Path):
                return obj.as_posix()
            elif isinstance(obj, dict):
                return type(obj)({k: _format_data(v) for k, v in obj.items()})
            elif isinstance(obj, (list, tuple)):
                return [_format_data(i) for i in obj]
            else:
                return obj

        return _format_data(self)

    @property
    def json(self):
        return self._to_json()

    def save_to_json(self, save_path, indent=4, ensure_ascii=False, *args, **kwargs):
        if not str(save_path).endswith(".json"):
            save_path = Path(save_path) / f"{Path(self['input_path']).stem}.json"
        _save_list_data(
            self._json_writer.write,
            save_path,
            self.json,
            indent=indent,
            ensure_ascii=ensure_ascii,
            *args,
            **kwargs,
        )


class Base64Mixin:
    def __init__(self, *args, **kwargs):
        self._base64_writer = TextWriter(*args, **kwargs)
        self._show_funcs.append(self.save_to_base64)

    @abstractmethod
    def _to_base64(self):
        raise NotImplementedError

    @property
    def base64(self):
        return self._to_base64()

    def save_to_base64(self, save_path, *args, **kwargs):
        if not str(save_path).lower().endswith((".b64")):
            fp = Path(self["input_path"])
            save_path = Path(save_path) / f"{fp.stem}{fp.suffix}"
        _save_list_data(
            self._base64_writer.write, save_path, self.base64, *args, **kwargs
        )


class ImgMixin:
    def __init__(self, backend="pillow", *args, **kwargs):
        self._img_writer = ImageWriter(backend=backend, *args, **kwargs)
        self._show_funcs.append(self.save_to_img)

    @abstractmethod
    def _to_img(self):
        raise NotImplementedError

    @property
    def img(self):
        image = self._to_img()
        # The img must be a PIL.Image obj
        if isinstance(image, np.ndarray):
            return Image.fromarray(image)
        return image

    def save_to_img(self, save_path, *args, **kwargs):
        if not str(save_path).lower().endswith((".jpg", ".png")):
            fp = Path(self["input_path"])
            save_path = Path(save_path) / f"{fp.stem}{fp.suffix}"
        _save_list_data(self._img_writer.write, save_path, self.img, *args, **kwargs)


class CSVMixin:
    def __init__(self, backend="pandas", *args, **kwargs):
        self._csv_writer = CSVWriter(backend=backend, *args, **kwargs)
        self._show_funcs.append(self.save_to_csv)

    @abstractmethod
    def _to_csv(self):
        raise NotImplementedError

    def save_to_csv(self, save_path, *args, **kwargs):
        if not str(save_path).endswith(".csv"):
            save_path = Path(save_path) / f"{Path(self['input_path']).stem}.csv"
        _save_list_data(
            self._csv_writer.write, save_path, self._to_csv(), *args, **kwargs
        )


class HtmlMixin:
    def __init__(self, *args, **kwargs):
        self._html_writer = HtmlWriter(*args, **kwargs)
        self._show_funcs.append(self.save_to_html)

    @property
    def html(self):
        return self._to_html()

    def _to_html(self):
        return self["html"]

    def save_to_html(self, save_path, *args, **kwargs):
        if not str(save_path).endswith(".html"):
            save_path = Path(save_path) / f"{Path(self['input_path']).stem}.html"
        _save_list_data(self._html_writer.write, save_path, self.html, *args, **kwargs)


class XlsxMixin:
    def __init__(self, *args, **kwargs):
        self._xlsx_writer = XlsxWriter(*args, **kwargs)
        self._show_funcs.append(self.save_to_xlsx)

    def _to_xlsx(self):
        return self["html"]

    def save_to_xlsx(self, save_path, *args, **kwargs):
        if not str(save_path).endswith(".xlsx"):
            save_path = Path(save_path) / f"{Path(self['input_path']).stem}.xlsx"
        _save_list_data(self._xlsx_writer.write, save_path, self.html, *args, **kwargs)


class MarkdownMixin:
    def __init__(self):
        self._markdown_writer = MarkdownWriter()
        self._show_funcs.append(self.save_to_markdown)

    def _to_markdown(self):
        def _format_data(obj):
            def format_title(content_value):
                content_value = content_value.rstrip('.')
                level = content_value.count('.') + 1 if '.' in content_value else 1
                return f"{'#' * level} {content_value}".replace('-\n', '').replace('\n', ' ')

            def format_centered_text(key):
                return f'<div style="text-align: center;">{sub_block[key]}</div>'.replace('-\n', '').replace('\n', ' ')+'\n'

            def format_image():
                img_tags = []
                if 'img' in sub_block['image']:
                    img_tags.append('<div style="text-align: center;"><img src="{}" alt="Image" /></div>'.format(
                        sub_block["image"]["img"].replace('-\n', '').replace('\n', ' '))
                    )
                if 'image_text' in sub_block['image']:
                    img_tags.append('<div style="text-align: center;">{}</div>'.format(
                        sub_block["image"]["image_text"].replace('-\n', '').replace('\n', ' '))
                    )
                return '\n'.join(img_tags)
            
            def format_chart():
                img_tags = []
                if 'img' in sub_block['chart']:
                    img_tags.append('<div style="text-align: center;"><img src="{}" alt="Image" /></div>'.format(
                        sub_block["chart"]["img"].replace('-\n', '').replace('\n', ' '))
                    )
                if 'image_text' in sub_block['chart']:
                    img_tags.append('<div style="text-align: center;">{}</div>'.format(
                        sub_block["chart"]["image_text"].replace('-\n', '').replace('\n', ' '))
                    )
                return '\n'.join(img_tags)

            def format_reference():
                pattern = r'\[\d+\]'
    # 替换匹配到的内容，在前面添加换行符
                res = re.sub(pattern, lambda match: '\n' + match.group(), sub_block['reference'])
                return "\n"+res

            def format_table():
                return "\n"+sub_block['table']
            
            handlers = {
                'paragraph_title': lambda: format_title(sub_block['paragraph_title']),
                # 'text_without_layout': lambda: format_title(sub_block['text_without_layout']),
                'doc_title': lambda: f"# {sub_block['doc_title']}".replace('-\n', '').replace('\n', ' '),
                'table_title': lambda: format_centered_text('table_title'),
                'figure_title': lambda: format_centered_text('figure_title'),
                'chart_title': lambda: format_centered_text('chart_title'),
                # 'text': lambda: sub_block['text'].replace('-\n', '').replace('\n', ' ').strip(),
                'text': lambda: sub_block['text'].strip('\n'),
                # 'number': lambda: str(sub_block['number']),
                'abstract': lambda: "\n"+sub_block['abstract'].strip('\n'),
                'content': lambda: sub_block['content'].replace('-\n', '').replace('\n', ' ').strip(),
                'image': format_image,
                'chart': format_chart,
                'formula': lambda: f"$${sub_block['formula']}$$".replace('-\n', '').replace('\n', ' '),
                'table': format_table,
                # 'reference': lambda: "\n"+f"**Reference**: {sub_block['reference']}".replace('-\n', '').replace('\n', ' ').replace('[','\n['),
                'reference': format_reference,
                'algorithm': lambda: "\n"+f"**Algorithm**: {sub_block['algorithm']}".replace('-\n', '').replace('\n', ' '),
                'seal': lambda: "\n"+f"**Seal**: {sub_block['seal']}".replace('-\n', '').replace('\n', ' '),
            }
            parsing_result = obj['layout_parsing_result']
            markdown_content = ""
            for block in parsing_result: # for each block show ordering results
                sub_blocks = block['sub_blocks']
                last_label = None
                seg_start_flag = None
                seg_end_flag = None
                for sub_block in sorted(sub_blocks, key=lambda x: x.get('sub_index',999)):
                    label = sub_block.get('label')
                    seg_start_flag = sub_block.get('seg_start_flag')
                    handler = handlers.get(label)
                    if handler:
                        if label == last_label == "text"  and  seg_start_flag == seg_end_flag == False:
                            markdown_content += " "+handler()
                        else:
                            markdown_content += "\n\n"+handler()
                        last_label = label
                        seg_end_flag = sub_block.get('seg_end_flag')

            return markdown_content
        return _format_data(self)

    @property
    def markdown(self):
        return self._to_markdown()

    def save_to_markdown(self, save_path, *args, **kwargs):
        if not str(save_path).endswith(".md"):
            save_path = Path(save_path) / f"{Path(self['input_path']).stem}.md"
        _save_list_data(self._markdown_writer.write, save_path, self.markdown, *args, **kwargs)