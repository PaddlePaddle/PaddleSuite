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
import ast
import math
from pathlib import Path
from copy import deepcopy

import numpy as np
import cv2

from .....utils.flags import (
    INFER_BENCHMARK,
    INFER_BENCHMARK_ITER,
    INFER_BENCHMARK_DATA_SIZE,
)
from ....utils.io import ImageReader, PDFReader
from . import funcs as F


class ReadImage:
    """Load image from the file."""

    _FLAGS_DICT = {
        "BGR": cv2.IMREAD_COLOR,
        "RGB": cv2.IMREAD_COLOR,
        "GRAY": cv2.IMREAD_GRAYSCALE,
    }

    def __init__(self, format="BGR"):
        """
        Initialize the instance.

        Args:
            format (str, optional): Target color format to convert the image to.
                Choices are 'BGR', 'RGB', and 'GRAY'. Default: 'BGR'.
        """
        super().__init__()
        self.format = format
        flags = self._FLAGS_DICT[self.format]
        self._img_reader = ImageReader(backend="opencv", flags=flags)

    def __call__(self, imgs):
        """apply"""
        return [self.read(img) for img in imgs]

    def read(self, img):
        if isinstance(img, np.ndarray):
            if self.format == "RGB":
                img = img[:, :, ::-1]
            return img
        elif isinstance(img, str):
            blob = self._img_reader.read(img)
            if blob is None:
                raise Exception(f"Image read Error: {img}")

            if self.format == "RGB":
                if blob.ndim != 3:
                    raise RuntimeError("Array is not 3-dimensional.")
                # BGR to RGB
                blob = blob[..., ::-1]
            return blob
        else:
            raise TypeError(
                f"ReadImage only supports the following types:\n"
                f"1. str, indicating a image file path or a directory containing image files.\n"
                f"2. numpy.ndarray.\n"
                f"However, got type: {type(img).__name__}."
            )


class _BaseResize:
    _INTERP_DICT = {
        "NEAREST": cv2.INTER_NEAREST,
        "LINEAR": cv2.INTER_LINEAR,
        "CUBIC": cv2.INTER_CUBIC,
        "AREA": cv2.INTER_AREA,
        "LANCZOS4": cv2.INTER_LANCZOS4,
    }

    def __init__(self, size_divisor, interp):
        super().__init__()

        if size_divisor is not None:
            assert isinstance(
                size_divisor, int
            ), "`size_divisor` should be None or int."
        self.size_divisor = size_divisor

        try:
            interp = self._INTERP_DICT[interp]
        except KeyError:
            raise ValueError(
                "`interp` should be one of {}.".format(self._INTERP_DICT.keys())
            )
        self.interp = interp

    @staticmethod
    def _rescale_size(img_size, target_size):
        """rescale size"""
        scale = min(max(target_size) / max(img_size), min(target_size) / min(img_size))
        rescaled_size = [round(i * scale) for i in img_size]
        return rescaled_size, scale


class Resize(_BaseResize):
    """Resize the image."""

    def __init__(
        self, target_size, keep_ratio=False, size_divisor=None, interp="LINEAR"
    ):
        """
        Initialize the instance.

        Args:
            target_size (list|tuple|int): Target width and height.
            keep_ratio (bool, optional): Whether to keep the aspect ratio of resized
                image. Default: False.
            size_divisor (int|None, optional): Divisor of resized image size.
                Default: None.
            interp (str, optional): Interpolation method. Choices are 'NEAREST',
                'LINEAR', 'CUBIC', 'AREA', and 'LANCZOS4'. Default: 'LINEAR'.
        """
        super().__init__(size_divisor=size_divisor, interp=interp)

        if isinstance(target_size, int):
            target_size = [target_size, target_size]
        F.check_image_size(target_size)
        self.target_size = target_size

        self.keep_ratio = keep_ratio

    def __call__(self, imgs):
        """apply"""
        return [self.resize(img) for img in imgs]

    def resize(self, img):
        target_size = self.target_size
        original_size = img.shape[:2][::-1]

        if self.keep_ratio:
            h, w = img.shape[0:2]
            target_size, _ = self._rescale_size((w, h), self.target_size)

        if self.size_divisor:
            target_size = [
                math.ceil(i / self.size_divisor) * self.size_divisor
                for i in target_size
            ]
        img = F.resize(img, target_size, interp=self.interp)
        return img


class ResizeByLong(_BaseResize):
    """
    Proportionally resize the image by specifying the target length of the
    longest side.
    """

    def __init__(self, target_long_edge, size_divisor=None, interp="LINEAR"):
        """
        Initialize the instance.

        Args:
            target_long_edge (int): Target length of the longest side of image.
            size_divisor (int|None, optional): Divisor of resized image size.
                Default: None.
            interp (str, optional): Interpolation method. Choices are 'NEAREST',
                'LINEAR', 'CUBIC', 'AREA', and 'LANCZOS4'. Default: 'LINEAR'.
        """
        super().__init__(size_divisor=size_divisor, interp=interp)
        self.target_long_edge = target_long_edge

    def __call__(self, imgs):
        """apply"""
        return [self.resize(img) for img in imgs]

    def resize(self, img):
        h, w = img.shape[:2]
        scale = self.target_long_edge / max(h, w)
        h_resize = round(h * scale)
        w_resize = round(w * scale)
        if self.size_divisor is not None:
            h_resize = math.ceil(h_resize / self.size_divisor) * self.size_divisor
            w_resize = math.ceil(w_resize / self.size_divisor) * self.size_divisor

        img = F.resize(img, (w_resize, h_resize), interp=self.interp)
        return img


class ResizeByShort(_BaseResize):
    """
    Proportionally resize the image by specifying the target length of the
    shortest side.
    """

    def __init__(self, target_short_edge, size_divisor=None, interp="LINEAR"):
        """
        Initialize the instance.

        Args:
            target_short_edge (int): Target length of the shortest side of image.
            size_divisor (int|None, optional): Divisor of resized image size.
                Default: None.
            interp (str, optional): Interpolation method. Choices are 'NEAREST',
                'LINEAR', 'CUBIC', 'AREA', and 'LANCZOS4'. Default: 'LINEAR'.
        """
        super().__init__(size_divisor=size_divisor, interp=interp)
        self.target_short_edge = target_short_edge

    def __call__(self, imgs):
        """apply"""
        return [self.resize(img) for img in imgs]

    def resize(self, img):
        h, w = img.shape[:2]
        scale = self.target_short_edge / min(h, w)
        h_resize = round(h * scale)
        w_resize = round(w * scale)
        if self.size_divisor is not None:
            h_resize = math.ceil(h_resize / self.size_divisor) * self.size_divisor
            w_resize = math.ceil(w_resize / self.size_divisor) * self.size_divisor

        img = F.resize(img, (w_resize, h_resize), interp=self.interp)
        return img


class Normalize:
    """Normalize the image."""

    def __init__(self, scale=1.0 / 255, mean=0.5, std=0.5, preserve_dtype=False):
        """
        Initialize the instance.

        Args:
            scale (float, optional): Scaling factor to apply to the image before
                applying normalization. Default: 1/255.
            mean (float|tuple|list, optional): Means for each channel of the image.
                Default: 0.5.
            std (float|tuple|list, optional): Standard deviations for each channel
                of the image. Default: 0.5.
            preserve_dtype (bool, optional): Whether to preserve the original dtype
                of the image.
        """
        super().__init__()

        self.scale = np.float32(scale)
        if isinstance(mean, float):
            mean = [mean]
        self.mean = np.asarray(mean).astype("float32")
        if isinstance(std, float):
            std = [std]
        self.std = np.asarray(std).astype("float32")
        self.preserve_dtype = preserve_dtype

    def __call__(self, imgs):
        """apply"""
        old_type = imgs[0].dtype
        # XXX: If `old_type` has higher precision than float32,
        # we will lose some precision.
        imgs = np.array(imgs).astype("float32", copy=False)
        imgs *= self.scale
        imgs -= self.mean
        imgs /= self.std
        if self.preserve_dtype:
            imgs = imgs.astype(old_type, copy=False)
        return list(imgs)


class ToCHWImage:
    """Reorder the dimensions of the image from HWC to CHW."""

    def __call__(self, imgs):
        """apply"""
        return [img.transpose((2, 0, 1)) for img in imgs]


class ToBatch:
    def __call__(self, imgs):
        return [np.stack(imgs, axis=0).astype(dtype=np.float32, copy=False)]
