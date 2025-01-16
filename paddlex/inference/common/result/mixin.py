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

from typing import Union, Tuple, List, Dict, Any, Iterator
from abc import abstractmethod
from pathlib import Path
import mimetypes
import json
import copy
import numpy as np
from PIL import Image
import pandas as pd

from ....utils import logging
from ...utils.io import (
    JsonWriter,
    ImageReader,
    ImageWriter,
    CSVWriter,
    HtmlWriter,
    XlsxWriter,
    TextWriter,
    VideoWriter,
    MarkdownWriter
)


class StrMixin:
    """Mixin class for adding string conversion capabilities."""

    @property
    def str(self) -> Dict[str, str]:
        """Property to get the string representation of the result.

        Returns:
            Dict[str, str]: The string representation of the result.
        """

        return self._to_str()

    def _to_str(
        self,
    ):
        """Convert the given result data to a string representation.

        Args:
            json_format (bool): If True, return a JSON formatted string. Default is False.
            indent (int): Number of spaces to indent for JSON formatting. Default is 4.
            ensure_ascii (bool): If True, ensure all characters are ASCII. Default is False.

        Returns:
            Dict[str, str]: The string representation of the result.
        """
        return {"res": str(self)}

    def print(self) -> None:
        """Print the string representation of the result."""
        logging.info(self.str)


class JsonMixin:
    """Mixin class for adding JSON serialization capabilities."""

    def __init__(self) -> None:
        self._json_writer = JsonWriter()
        self._save_funcs.append(self.save_to_json)

    def _to_json(self) -> Dict[str, Dict[str, Any]]:
        """Convert the object to a JSON-serializable format.

        Returns:
            Dict[str, Dict[str, Any]]: A dictionary representation of the object that is JSON-serializable.
        """

        def _format_data(obj):
            """Helper function to format data into a JSON-serializable format.

            Args:
                obj: The object to be formatted.

            Returns:
                Any: The formatted object.
            """
            if isinstance(obj, np.float32):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return [_format_data(item) for item in obj.tolist()]
            elif isinstance(obj, pd.DataFrame):
                return obj.to_json(orient="records", force_ascii=False)
            elif isinstance(obj, Path):
                return obj.as_posix()
            elif isinstance(obj, dict):
                return dict({k: _format_data(v) for k, v in obj.items()})
            elif isinstance(obj, (list, tuple)):
                return [_format_data(i) for i in obj]
            else:
                return obj

        return {"res": _format_data(copy.deepcopy(self))}

    @property
    def json(self) -> Dict[str, Dict[str, Any]]:
        """Property to get the JSON representation of the result.

        Returns:
            Dict[str, Dict[str, Any]]: The dict type JSON representation of the result.
        """

        return self._to_json()

    def save_to_json(
        self,
        save_path: str,
        indent: int = 4,
        ensure_ascii: bool = False,
        *args: List,
        **kwargs: Dict,
    ) -> None:
        """Save the JSON representation of the object to a file.

        Args:
            save_path (str): The path to save the JSON file. If the save path does not end with '.json', it appends the base name and suffix of the input path.
            indent (int): The number of spaces to indent for pretty printing. Default is 4.
            ensure_ascii (bool): If False, non-ASCII characters will be included in the output. Default is False.
            *args: Additional positional arguments to pass to the underlying writer.
            **kwargs: Additional keyword arguments to pass to the underlying writer.
        """

        def _is_json_file(file_path):
            mime_type, _ = mimetypes.guess_type(file_path)
            return mime_type is not None and mime_type == "application/json"

        if not _is_json_file(save_path):
            fp = Path(self["input_path"])
            stem = fp.stem
            suffix = fp.suffix
            base_save_path = Path(save_path)
            for key in self.json:
                save_path = base_save_path / f"{stem}_{key}.json"
                self._json_writer.write(
                    save_path.as_posix(), self.json[key], *args, **kwargs
                )
        else:
            if len(self.json) > 1:
                logging.warning(
                    f"The result has multiple json files need to be saved. But the `save_path` has been specfied as `{save_path}`!"
                )
            self._json_writer.write(
                save_path,
                self.json[list(self.json.keys())[0]],
                indent=indent,
                ensure_ascii=ensure_ascii,
                *args,
                **kwargs,
            )


class Base64Mixin:
    """Mixin class for adding Base64 encoding capabilities."""

    def __init__(self, *args: List, **kwargs: Dict) -> None:
        """Initializes the Base64Mixin.

        Args:
            *args: Positional arguments to pass to the TextWriter.
            **kwargs: Keyword arguments to pass to the TextWriter.
        """
        self._base64_writer = TextWriter(*args, **kwargs)
        self._save_funcs.append(self.save_to_base64)

    @abstractmethod
    def _to_base64(self) -> Dict[str, str]:
        """Abstract method to convert the result to Base64.

        Returns:
            Dict[str, str]: The str type Base64 representation result.
        """
        raise NotImplementedError

    @property
    def base64(self) -> Dict[str, str]:
        """
        Property that returns the Base64 encoded content.

        Returns:
            Dict[str, str]: The base64 representation of the result.
        """
        return self._to_base64()

    def save_to_base64(self, save_path: str, *args: List, **kwargs: Dict) -> None:
        """Saves the Base64 encoded content to the specified path.

        Args:
            save_path (str): The path to save the base64 representation result. If the save path does not end with '.b64', it appends the base name and suffix of the input path.

            *args: Additional positional arguments that will be passed to the base64 writer.
            **kwargs: Additional keyword arguments that will be passed to the base64 writer.
        """
        if not str(save_path).lower().endswith((".b64")):
            fp = Path(self["input_path"])
            stem = fp.stem
            suffix = fp.suffix
            base_save_path = Path(save_path)
            for key in self.base64:
                save_path = base_save_path / f"{stem}_{key}.b64"
                self._base64_writer.write(
                    save_path.as_posix(), self.base64[key], *args, **kwargs
                )
        else:
            if len(self.base64) > 1:
                logging.warning(
                    f"The result has multiple base64 files need to be saved. But the `save_path` has been specfied as `{save_path}`!"
                )
            self._base64_writer.write(
                save_path, self.base64[list(self.base64.keys())[0]], *args, **kwargs
            )


class ImgMixin:
    """Mixin class for adding image handling capabilities."""

    def __init__(self, backend: str = "pillow", *args: List, **kwargs: Dict) -> None:
        """Initializes ImgMixin.

        Args:
            backend (str): The backend to use for image processing. Defaults to "pillow".
            *args: Additional positional arguments to pass to the ImageWriter.
            **kwargs: Additional keyword arguments to pass to the ImageWriter.
        """
        self._img_writer = ImageWriter(backend=backend, *args, **kwargs)
        self._save_funcs.append(self.save_to_img)

    @abstractmethod
    def _to_img(self) -> Dict[str, Image.Image]:
        """Abstract method to convert the result to an image.

        Returns:
            Dict[str, Image.Image]: The image representation result.
        """
        raise NotImplementedError

    @property
    def img(self) -> Dict[str, Image.Image]:
        """Property to get the image representation of the result.

        Returns:
            Dict[str, Image.Image]: The image representation of the result.
        """
        return self._to_img()

    def save_to_img(self, save_path: str, *args: List, **kwargs: Dict) -> None:
        """Saves the image representation of the result to the specified path.

        Args:
            save_path (str): The path to save the image. If the save path does not end with .jpg or .png, it appends the input path's stem and suffix to the save path.
            *args: Additional positional arguments that will be passed to the image writer.
            **kwargs: Additional keyword arguments that will be passed to the image writer.
        """

        def _is_image_file(file_path):
            mime_type, _ = mimetypes.guess_type(file_path)
            return mime_type is not None and mime_type.startswith("image/")

        if not _is_image_file(save_path):
            fp = Path(self["input_path"])
            stem = fp.stem
            suffix = fp.suffix
            base_save_path = Path(save_path)
            for key in self.img:
                save_path = base_save_path / f"{stem}_{key}{suffix}"
                self._img_writer.write(
                    save_path.as_posix(), self.img[key], *args, **kwargs
                )
        else:
            if len(self.img) > 1:
                logging.warning(
                    f"The result has multiple img files need to be saved. But the `save_path` has been specfied as `{save_path}`!"
                )
            self._img_writer.write(
                save_path, self.img[list(self.img.keys())[0]], *args, **kwargs
            )


class CSVMixin:
    """Mixin class for adding CSV handling capabilities."""

    def __init__(self, backend: str = "pandas", *args: List, **kwargs: Dict) -> None:
        """Initializes the CSVMixin.

        Args:
            backend (str): The backend to use for CSV operations (default is "pandas").
            *args: Optional positional arguments to pass to the CSVWriter.
            **kwargs: Optional keyword arguments to pass to the CSVWriter.
        """
        self._csv_writer = CSVWriter(backend=backend, *args, **kwargs)
        if not hasattr(self, "_save_funcs"):
            self._save_funcs = []
        self._save_funcs.append(self.save_to_csv)

    @property
    def csv(self) -> Dict[str, pd.DataFrame]:
        """Property to get the pandas Dataframe representation of the result.

        Returns:
            Dict[str, pd.DataFrame]: The pandas.DataFrame representation of the result.
        """
        return self._to_csv()

    @abstractmethod
    def _to_csv(self) -> Dict[str, pd.DataFrame]:
        """Abstract method to convert the result to pandas.DataFrame.

        Returns:
            Dict[str, pd.DataFrame]: The pandas.DataFrame representation result.
        """
        raise NotImplementedError

    def save_to_csv(self, save_path: str, *args: List, **kwargs: Dict) -> None:
        """Saves the result to a CSV file.

        Args:
            save_path (str): The path to save the CSV file. If the path does not end with ".csv",
                the stem of the input path attribute (self['input_path']) will be used as the filename.
            *args: Optional positional arguments to pass to the CSV writer's write method.
            **kwargs: Optional keyword arguments to pass to the CSV writer's write method.
        """

        def _is_csv_file(file_path):
            mime_type, _ = mimetypes.guess_type(file_path)
            return mime_type is not None and mime_type == "text/csv"

        if not _is_csv_file(save_path):
            fp = Path(self["input_path"])
            stem = fp.stem
            base_save_path = Path(save_path)
            for key in self.csv:
                save_path = base_save_path / f"{stem}_{key}.csv"
                self._csv_writer.write(
                    save_path.as_posix(), self.csv[key], *args, **kwargs
                )
        else:
            if len(self.csv) > 1:
                logging.warning(
                    f"The result has multiple csv files need to be saved. But the `save_path` has been specfied as `{save_path}`!"
                )
            self._csv_writer.write(
                save_path, self.csv[list(self.csv.keys())[0]], *args, **kwargs
            )


class HtmlMixin:
    """Mixin class for adding HTML handling capabilities."""

    def __init__(self, *args: List, **kwargs: Dict) -> None:
        """
        Initializes the HTML writer and appends the save_to_html method to the save functions list.

        Args:
            *args: Positional arguments passed to the HtmlWriter.
            **kwargs: Keyword arguments passed to the HtmlWriter.
        """
        self._html_writer = HtmlWriter(*args, **kwargs)
        self._save_funcs.append(self.save_to_html)

    @property
    def html(self) -> Dict[str, str]:
        """Property to get the HTML representation of the result.

        Returns:
            str: The str type HTML representation of the result.
        """
        return self._to_html()

    @abstractmethod
    def _to_html(self) -> Dict[str, str]:
        """Abstract method to convert the result to str type HTML representation.

        Returns:
            Dict[str, str]: The str type HTML representation result.
        """
        raise NotImplementedError

    def save_to_html(self, save_path: str, *args: List, **kwargs: Dict) -> None:
        """Saves the HTML representation of the object to the specified path.

        Args:
            save_path (str): The path to save the HTML file.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """

        def _is_html_file(file_path):
            mime_type, _ = mimetypes.guess_type(file_path)
            return mime_type is not None and mime_type == "text/html"

        if not _is_html_file(save_path):
            fp = Path(self["input_path"])
            stem = fp.stem
            base_save_path = Path(save_path)
            for key in self.html:
                save_path = base_save_path / f"{stem}_{key}.html"
                self._html_writer.write(
                    save_path.as_posix(), self.html[key], *args, **kwargs
                )
        else:
            if len(self.html) > 1:
                logging.warning(
                    f"The result has multiple html files need to be saved. But the `save_path` has been specfied as `{save_path}`!"
                )
            self._html_writer.write(
                save_path, self.html[list(self.html.keys())[0]], *args, **kwargs
            )


class XlsxMixin:
    """Mixin class for adding XLSX handling capabilities."""

    def __init__(self, *args: List, **kwargs: Dict) -> None:
        """Initializes the XLSX writer and appends the save_to_xlsx method to the save functions.

        Args:
            *args: Positional arguments to be passed to the XlsxWriter constructor.
            **kwargs: Keyword arguments to be passed to the XlsxWriter constructor.
        """
        self._xlsx_writer = XlsxWriter(*args, **kwargs)
        self._save_funcs.append(self.save_to_xlsx)

    @property
    def xlsx(self) -> Dict[str, str]:
        """Property to get the XLSX representation of the result.

        Returns:
            Dict[str, str]: The str type XLSX representation of the result.
        """
        return self._to_xlsx()

    @abstractmethod
    def _to_xlsx(self) -> Dict[str, str]:
        """Abstract method to convert the result to str type XLSX representation.

        Returns:
            Dict[str, str]: The str type HTML representation result.
        """
        raise NotImplementedError

    def save_to_xlsx(self, save_path: str, *args: List, **kwargs: Dict) -> None:
        """Saves the HTML representation to an XLSX file.

        Args:
            save_path (str): The path to save the XLSX file. If the path does not end with ".xlsx",
                             the filename will be set to the stem of the input path with ".xlsx" extension.
            *args: Additional positional arguments to pass to the XLSX writer.
            **kwargs: Additional keyword arguments to pass to the XLSX writer.
        """

        def _is_xlsx_file(file_path):
            mime_type, _ = mimetypes.guess_type(file_path)
            return (
                mime_type is not None
                and mime_type
                == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        if not _is_xlsx_file(save_path):
            fp = Path(self["input_path"])
            stem = fp.stem
            base_save_path = Path(save_path)
            for key in self.xlsx:
                save_path = base_save_path / f"{stem}_{key}.xlsx"
                self._xlsx_writer.write(
                    save_path.as_posix(), self.xlsx[key], *args, **kwargs
                )
        else:
            if len(self.xlsx) > 1:
                logging.warning(
                    f"The result has multiple xlsx files need to be saved. But the `save_path` has been specfied as `{save_path}`!"
                )
            self._xlsx_writer.write(
                save_path, self.xlsx[list(self.xlsx.keys())[0]], *args, **kwargs
            )


class VideoMixin:
    """Mixin class for adding Video handling capabilities."""

    def __init__(self, backend: str = "opencv", *args: List, **kwargs: Dict) -> None:
        """Initializes VideoMixin.

        Args:
            backend (str): The backend to use for video processing. Defaults to "opencv".
            *args: Additional positional arguments to pass to the VideoWriter.
            **kwargs: Additional keyword arguments to pass to the VideoWriter.
        """
        self._backend = backend
        self._save_funcs.append(self.save_to_video)

    @abstractmethod
    def _to_video(self) -> Dict[str, np.array]:
        """Abstract method to convert the result to a video.

        Returns:
            Dict[str, np.array]: The video representation result.
        """
        raise NotImplementedError

    @property
    def video(self) -> Dict[str, np.array]:
        """Property to get the video representation of the result.

        Returns:
            Dict[str, np.array]: The video representation of the result.
        """
        return self._to_video()

    def save_to_video(self, save_path: str, *args: List, **kwargs: Dict) -> None:
        """Saves the video representation of the result to the specified path.

        Args:
            save_path (str): The path to save the video. If the save path does not end with .mp4 or .avi, it appends the input path's stem and suffix to the save path.
            *args: Additional positional arguments that will be passed to the video writer.
            **kwargs: Additional keyword arguments that will be passed to the video writer.
        """

        def _is_video_file(file_path):
            mime_type, _ = mimetypes.guess_type(file_path)
            return mime_type is not None and mime_type.startswith("video/")

        video_writer = VideoWriter(backend=self._backend, *args, **kwargs)

        if not _is_video_file(save_path):
            fp = Path(self["input_path"])
            stem = fp.stem
            suffix = fp.suffix
            base_save_path = Path(save_path)
            for key in self.video:
                save_path = base_save_path / f"{stem}_{key}{suffix}"
                video_writer.write(
                    save_path.as_posix(), self.video[key], *args, **kwargs
                )
        else:
            if len(self.video) > 1:
                logging.warning(
                    f"The result has multiple video files need to be saved. But the `save_path` has been specfied as `{save_path}`!"
                )
            video_writer.write(
                save_path, self.video[list(self.video.keys())[0]], *args, **kwargs
            )

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
        self._save_list_data(self._markdown_writer.write, save_path, self.markdown, *args, **kwargs)
    
    def _save_list_data(self,save_func, save_path, data, *args, **kwargs):
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