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


import os
import enum

import cv2
import pandas as pd
import numpy as np
from PIL import Image

__all__ = ["ImageWriter", "TextWriter", "WriterType", "TSWriter"]


class WriterType(enum.Enum):
    """WriterType"""

    IMAGE = 1
    VIDEO = 2
    TEXT = 3
    TS = 4


class _BaseWriter(object):
    """_BaseWriter"""

    def __init__(self, backend, **bk_args):
        super().__init__()
        if len(bk_args) == 0:
            bk_args = self.get_default_backend_args()
        self.bk_type = backend
        self.bk_args = bk_args
        self._backend = self.get_backend()

    def write(self, out_path, obj):
        """write"""
        raise NotImplementedError

    def get_backend(self, bk_args=None):
        """get backend"""
        if bk_args is None:
            bk_args = self.bk_args
        return self._init_backend(self.bk_type, bk_args)

    def _init_backend(self, bk_type, bk_args):
        """init backend"""
        raise NotImplementedError

    def get_type(self):
        """get type"""
        raise NotImplementedError

    def get_default_backend_args(self):
        """get default backend arguments"""
        return {}


class ImageWriter(_BaseWriter):
    """ImageWriter"""

    def __init__(self, backend="opencv", **bk_args):
        super().__init__(backend=backend, **bk_args)

    def write(self, out_path, obj):
        """write"""
        return self._backend.write_obj(out_path, obj)

    def _init_backend(self, bk_type, bk_args):
        """init backend"""
        if bk_type == "opencv":
            return OpenCVImageWriterBackend(**bk_args)
        elif bk_type == "pillow":
            return PILImageWriterBackend(**bk_args)
        else:
            raise ValueError("Unsupported backend type")

    def get_type(self):
        """get type"""
        return WriterType.IMAGE


class TextWriter(_BaseWriter):
    """TextWriter"""

    def __init__(self, backend="python", **bk_args):
        super().__init__(backend=backend, **bk_args)

    def write(self, out_path, obj):
        """write"""
        return self._backend.write_obj(out_path, obj)

    def _init_backend(self, bk_type, bk_args):
        """init backend"""
        if bk_type == "python":
            return TextWriterBackend(**bk_args)
        else:
            raise ValueError("Unsupported backend type")

    def get_type(self):
        """get type"""
        return WriterType.TEXT


class _BaseWriterBackend(object):
    """_BaseWriterBackend"""

    def write_obj(self, out_path, obj):
        """write object"""
        out_dir = os.path.dirname(out_path)
        os.makedirs(out_dir, exist_ok=True)
        return self._write_obj(out_path, obj)

    def _write_obj(self, out_path, obj):
        """write object"""
        raise NotImplementedError


class TextWriterBackend(_BaseWriterBackend):
    """TextWriterBackend"""

    def __init__(self, mode="w", encoding="utf-8"):
        super().__init__()
        self.mode = mode
        self.encoding = encoding

    def _write_obj(self, out_path, obj):
        """write text object"""
        with open(out_path, mode=self.mode, encoding=self.encoding) as f:
            f.write(obj)


class _ImageWriterBackend(_BaseWriterBackend):
    """_ImageWriterBackend"""

    pass


class OpenCVImageWriterBackend(_ImageWriterBackend):
    """OpenCVImageWriterBackend"""

    def _write_obj(self, out_path, obj):
        """write image object by OpenCV"""
        if isinstance(obj, Image.Image):
            arr = np.asarray(obj)
        elif isinstance(obj, np.ndarray):
            arr = obj
        else:
            raise TypeError("Unsupported object type")
        return cv2.imwrite(out_path, arr)


class PILImageWriterBackend(_ImageWriterBackend):
    """PILImageWriterBackend"""

    def __init__(self, format_=None):
        super().__init__()
        self.format = format_

    def _write_obj(self, out_path, obj):
        """write image object by PIL"""
        if isinstance(obj, Image.Image):
            img = obj
        elif isinstance(obj, np.ndarray):
            img = Image.fromarray(obj)
        else:
            raise TypeError("Unsupported object type")
        return img.save(out_path, format=self.format)


class TSWriter(_BaseWriter):
    """TSWriter"""

    def __init__(self, backend="pandas", **bk_args):
        super().__init__(backend=backend, **bk_args)

    def write(self, out_path, obj):
        """write"""
        return self._backend.write_obj(out_path, obj)

    def _init_backend(self, bk_type, bk_args):
        """init backend"""
        if bk_type == "pandas":
            return PandasTSWriterBackend(**bk_args)
        else:
            raise ValueError("Unsupported backend type")

    def get_type(self):
        """get type"""
        return WriterType.TS


class _TSWriterBackend(_BaseWriterBackend):
    """_TSWriterBackend"""

    pass


class PandasTSWriterBackend(_TSWriterBackend):
    """PILImageWriterBackend"""

    def __init__(self):
        super().__init__()

    def _write_obj(self, out_path, obj):
        """write image object by PIL"""
        if isinstance(obj, pd.DataFrame):
            ts = obj
        else:
            raise TypeError("Unsupported object type")
        return ts.to_csv(out_path)
