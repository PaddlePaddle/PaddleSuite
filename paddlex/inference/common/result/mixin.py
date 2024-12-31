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
)


def _save_list_data(save_func, save_path, data, *args, **kwargs):
    """
    Save list type data to the specified path.
    If data type is a list, iterate through it and save each element using save_func with a modified filename (appending an index and the original file extension).

    Args:
        save_func (Callable): The function to be used for saving data.
        save_path (Union[str, Path]): The path to save the data.
        data (Union[None, list, Any]): The data to be saved. If None, the function will return immediately.
        *args: Additional positional arguments to be passed to save_func.
        **kwargs: Additional keyword arguments to be passed to save_func.

    Returns:
        None
    """
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
    """Mixin class for adding string conversion capabilities."""

    @property
    def str(self) -> str:
        """Property to get the string representation of the result.

        Returns:
            str: The str type string representation of the result.
        """

        return self._to_str(self)

    def _to_str(
        self,
        data: dict,
        json_format: bool = False,
        indent: int = 4,
        ensure_ascii: bool = False,
    ) -> str:
        """Convert the given result data to a string representation.

        Args:
            data (dict): The data would be converted to str.
            json_format (bool): If True, return a JSON formatted string. Default is False.
            indent (int): Number of spaces to indent for JSON formatting. Default is 4.
            ensure_ascii (bool): If True, ensure all characters are ASCII. Default is False.

        Returns:
            str: The string representation of the data.
        """
        if json_format:
            return json.dumps(data.json, indent=indent, ensure_ascii=ensure_ascii)
        else:
            return str(data)

    def print(
        self, json_format: bool = False, indent: int = 4, ensure_ascii: bool = False
    ) -> None:
        """Print the string representation of the result.

        Args:
            json_format (bool): If True, print a JSON formatted string. Default is False.
            indent (int): Number of spaces to indent for JSON formatting. Default is 4.
            ensure_ascii (bool): If True, ensure all characters are ASCII. Default is False.
        """
        str_ = self._to_str(
            self, json_format=json_format, indent=indent, ensure_ascii=ensure_ascii
        )
        logging.info(str_)


class JsonMixin:
    """Mixin class for adding JSON serialization capabilities."""

    def __init__(self) -> None:
        self._json_writer = JsonWriter()
        self._save_funcs.append(self.save_to_json)

    def _to_json(self) -> Dict[str, Any]:
        """Convert the object to a JSON-serializable format.

        Returns:
            Dict[str, Any]: A dictionary representation of the object that is JSON-serializable.
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

        return _format_data(copy.deepcopy(self))

    @property
    def json(self) -> Dict[str, Any]:
        """Property to get the JSON representation of the result.

        Returns:
            Dict[str, Any]: The dict type JSON representation of the result.
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
            save_path = Path(save_path) / f"{Path(self['input_path']).stem}.json"
            save_path = save_path.as_posix()
        self._json_writer.write(
            save_path,
            self.json,
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
    def _to_base64(self) -> str:
        """Abstract method to convert the result to Base64.

        Returns:
        str: The str type Base64 representation result.
        """
        raise NotImplementedError

    @property
    def base64(self) -> str:
        """
        Property that returns the Base64 encoded content.

        Returns:
            str: The base64 representation of the result.
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
            save_path = Path(save_path) / f"{fp.stem}{fp.suffix}"
        self._base64_writer.write(save_path.as_posix(), self.base64, *args, **kwargs)


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
    def _to_img(self) -> Union[np.ndarray, Image.Image]:
        """Abstract method to convert the result to an image.

        Returns:
        Union[np.ndarray, Image.Image]: The image representation result.
        """
        raise NotImplementedError

    @property
    def img(self) -> Image.Image:
        """Property to get the image representation of the result.

        Returns:
            Image.Image: The image representation of the result.
        """
        image = self._to_img()
        # The img must be a PIL.Image obj
        if isinstance(image, np.ndarray):
            return Image.fromarray(image)
        return image

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
            save_path = Path(save_path) / f"{fp.stem}{fp.suffix}"
            save_path = save_path.as_posix()
        self._img_writer.write(save_path, self.img, *args, **kwargs)


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
    def csv(self) -> pd.DataFrame:
        """Property to get the pandas Dataframe representation of the result.

        Returns:
            pandas.DataFrame: The pandas.DataFrame representation of the result.
        """
        return self._to_csv()

    @abstractmethod
    def _to_csv(self) -> pd.DataFrame:
        """Abstract method to convert the result to pandas.DataFrame.

        Returns:
        pandas.DataFrame: The pandas.DataFrame representation result.
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
        if not str(save_path).endswith(".csv"):
            save_path = Path(save_path) / f"{Path(self['input_path']).stem}.csv"
        self._csv_writer.write(save_path.as_posix(), self.csv, *args, **kwargs)


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
    def html(self) -> str:
        """Property to get the HTML representation of the result.

        Returns:
            str: The str type HTML representation of the result.
        """
        return self._to_html()

    @abstractmethod
    def _to_html(self) -> str:
        """Abstract method to convert the result to str type HTML representation.

        Returns:
        str: The str type HTML representation result.
        """
        raise NotImplementedError

    def save_to_html(self, save_path: str, *args: List, **kwargs: Dict) -> None:
        """Saves the HTML representation of the object to the specified path.

        Args:
            save_path (str): The path to save the HTML file.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        if not str(save_path).endswith(".html"):
            save_path = Path(save_path) / f"{Path(self['input_path']).stem}.html"
        self._html_writer.write(save_path.as_posix(), self.html, *args, **kwargs)


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
    def xlsx(self) -> str:
        """Property to get the XLSX representation of the result.

        Returns:
            str: The str type XLSX representation of the result.
        """
        return self._to_xlsx()

    @abstractmethod
    def _to_xlsx(self) -> str:
        """Abstract method to convert the result to str type XLSX representation.

        Returns:
        str: The str type HTML representation result.
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
        if not str(save_path).endswith(".xlsx"):
            save_path = Path(save_path) / f"{Path(self['input_path']).stem}.xlsx"
        self._xlsx_writer.write(save_path.as_posix(), self.xlsx, *args, **kwargs)


class VideoMixin:
    def __init__(self, backend="opencv", *args, **kwargs):
        self._video_writer = VideoWriter(backend=backend, *args, **kwargs)
        self._save_funcs.append(self.save_to_video)

    @abstractmethod
    def _to_video(self):
        raise NotImplementedError

    @property
    def video(self):
        video = self._to_video()
        return video

    def save_to_video(self, save_path, *args, **kwargs):
        if not str(save_path).lower().endswith((".mp4", ".avi", ".mkv")):
            fp = Path(self["input_path"])
            save_path = Path(save_path) / f"{fp.stem}{fp.suffix}"
        _save_list_data(
            self._video_writer.write, save_path, self.video, *args, **kwargs
        )
