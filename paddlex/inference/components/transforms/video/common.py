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

import ast
import math
from pathlib import Path
from copy import deepcopy

import lazy_paddle
import numpy as np
import cv2
from PIL import Image

from .....utils.flags import (
    INFER_BENCHMARK,
    INFER_BENCHMARK_ITER,
    INFER_BENCHMARK_DATA_SIZE,
)
from .....utils.cache import CACHE_DIR, temp_file_manager
from ....utils.io import VideoReader
from ...base import BaseComponent
from ..read_data import _BaseRead

__all__ = [
    "ReadVideo",
    "Scale",
    "CenterCrop",
    "Image2Array",
    "NormalizeVideo",
]


def _check_image_size(input_):
    """check image size"""
    if not (
        isinstance(input_, (list, tuple))
        and len(input_) == 2
        and isinstance(input_[0], int)
        and isinstance(input_[1], int)
    ):
        raise TypeError(f"{input_} cannot represent a valid image size.")


class ReadVideo(_BaseRead):
    """Load video from the file."""

    INPUT_KEYS = ["video"]
    OUTPUT_KEYS = ["video"]
    DEAULT_INPUTS = {"video": "video"}
    DEAULT_OUTPUTS = {
        "video": "video",
        "input_path": "input_path",
    }

    SUFFIX = ["mp4", "avi"]

    def __init__(
        self, batch_size=1, backend="opencv", num_seg=8, seg_len=1, sample_type=None
    ):
        """
        Initialize the instance.

        Args:
            format (str, optional): Target color format to convert the image to.
                Choices are 'BGR', 'RGB', and 'GRAY'. Default: 'BGR'.
        """
        super().__init__(batch_size)
        self._video_reader = VideoReader(
            backend=backend, num_seg=num_seg, seg_len=seg_len, sample_type=sample_type
        )

    def apply(self, video):
        """apply"""
        if isinstance(video, str):
            file_path = video
            file_path = self._download_from_url(file_path)
            file_list = self._get_files_list(file_path)
            batch = []
            for file_path in file_list:
                video = self._read(file_path)
                batch.extend(video)
                if len(batch) >= self.batch_size:
                    yield batch
                    batch = []
            if len(batch) > 0:
                yield batch
        else:
            raise TypeError(
                f"ReadVideo only supports the following types:\n"
                f"1. str, indicating a image file path or a directory containing image files.\n"
                f"However, got type: {type(video).__name__}."
            )

    def _read(self, file_path):
        return self._read_video(file_path)

    def _read_video(self, video_path):
        blob = list(self._video_reader.read(video_path))
        if blob is None:
            raise Exception("Video read Error")
        return [
            {
                "input_path": video_path,
                "video": blob,
            }
        ]


class Scale(BaseComponent):
    """Scale images."""

    INPUT_KEYS = ["video"]
    OUTPUT_KEYS = ["imgs", "video"]
    DEAULT_INPUTS = {"video": "video"}
    DEAULT_OUTPUTS = {
        "imgs": "imgs",
    }

    def __init__(
        self,
        short_size,
        fixed_ratio=True,
        keep_ratio=None,
        do_round=False,
        backend="pillow",
    ):
        super().__init__()
        self.short_size = short_size
        assert (fixed_ratio and not keep_ratio) or (
            not fixed_ratio
        ), f"fixed_ratio and keep_ratio cannot be true at the same time"
        self.fixed_ratio = fixed_ratio
        self.keep_ratio = keep_ratio
        self.do_round = do_round
        self.backend = backend

    def apply(self, video):
        """
        Performs resize operations.
        Args:
            imgs (Sequence[PIL.Image]): List where each item is a PIL.Image.
            For example, [PIL.Image0, PIL.Image1, PIL.Image2, ...]
        return:
            resized_imgs: List where each item is a PIL.Image after scaling.
        """

        imgs = video

        resized_imgs = []
        for i in range(len(imgs)):
            img = imgs[i]
            if isinstance(img, np.ndarray):
                h, w, _ = img.shape
            elif isinstance(img, Image.Image):
                w, h = img.size
            else:
                raise NotImplementedError
            if (w <= h and w == self.short_size) or (h <= w and h == self.short_size):
                if self.backend == "pillow" and not isinstance(img, Image.Image):
                    img = Image.fromarray(img)
                resized_imgs.append(img)
                continue

            if w <= h:
                ow = self.short_size
                if self.fixed_ratio:
                    oh = int(self.short_size * 4.0 / 3.0)
                elif self.keep_ratio is False:
                    oh = self.short_size
                else:
                    scale_factor = self.short_size / w
                    oh = (
                        int(h * float(scale_factor) + 0.5)
                        if self.do_round
                        else int(h * self.short_size / w)
                    )
                    ow = (
                        int(w * float(scale_factor) + 0.5)
                        if self.do_round
                        else self.short_size
                    )
            else:
                oh = self.short_size
                if self.fixed_ratio:
                    ow = int(self.short_size * 4.0 / 3.0)
                elif self.keep_ratio is False:
                    ow = self.short_size
                else:
                    scale_factor = self.short_size / h
                    oh = (
                        int(h * float(scale_factor) + 0.5)
                        if self.do_round
                        else self.short_size
                    )
                    ow = (
                        int(w * float(scale_factor) + 0.5)
                        if self.do_round
                        else int(w * self.short_size / h)
                    )
            if self.backend == "pillow":
                resized_imgs.append(img.resize((ow, oh), Image.BILINEAR))
            elif self.backend == "cv2" and (self.keep_ratio is not None):
                resized_imgs.append(
                    cv2.resize(img, (ow, oh), interpolation=cv2.INTER_LINEAR)
                )
            else:
                resized_imgs.append(
                    Image.fromarray(
                        cv2.resize(
                            np.asarray(img), (ow, oh), interpolation=cv2.INTER_LINEAR
                        )
                    )
                )
        imgs = resized_imgs
        return {
            "imgs": imgs,
        }


class CenterCrop(BaseComponent):
    """Center crop images."""

    INPUT_KEYS = ["imgs"]
    OUTPUT_KEYS = ["imgs", "imgs"]
    DEAULT_INPUTS = {"imgs": "imgs"}
    DEAULT_OUTPUTS = {
        "imgs": "imgs",
    }

    def __init__(self, target_size, do_round=True, backend="pillow"):
        super().__init__()
        self.target_size = target_size
        self.do_round = do_round
        self.backend = backend

    def apply(self, imgs):
        """
        Performs Center crop operations.
        Args:
            imgs: List where each item is a PIL.Image.
            For example, [PIL.Image0, PIL.Image1, PIL.Image2, ...]
        return:
            crop_imgs: List where each item is a PIL.Image after Center crop.
        """

        crop_imgs = []
        th, tw = self.target_size, self.target_size
        if isinstance(imgs, lazy_paddle.Tensor):
            h, w = imgs.shape[-2:]
            x1 = int(round((w - tw) / 2.0)) if self.do_round else (w - tw) // 2
            y1 = int(round((h - th) / 2.0)) if self.do_round else (h - th) // 2
            crop_imgs = imgs[:, :, y1 : y1 + th, x1 : x1 + tw]
        else:
            for img in imgs:
                if self.backend == "pillow":
                    w, h = img.size
                elif self.backend == "cv2":
                    h, w, _ = img.shape
                else:
                    raise NotImplementedError
                assert (w >= self.target_size) and (
                    h >= self.target_size
                ), "image width({}) and height({}) should be larger than crop size".format(
                    w, h, self.target_size
                )
                x1 = int(round((w - tw) / 2.0)) if self.do_round else (w - tw) // 2
                y1 = int(round((h - th) / 2.0)) if self.do_round else (h - th) // 2
                if self.backend == "cv2":
                    crop_imgs.append(img[y1 : y1 + th, x1 : x1 + tw])
                elif self.backend == "pillow":
                    crop_imgs.append(img.crop((x1, y1, x1 + tw, y1 + th)))
        return {
            "imgs": crop_imgs,
        }


class Image2Array(BaseComponent):
    """Image2Array"""

    INPUT_KEYS = ["imgs"]
    OUTPUT_KEYS = ["imgs", "imgs"]
    DEAULT_INPUTS = {"imgs": "imgs"}
    DEAULT_OUTPUTS = {
        "imgs": "imgs",
    }

    def __init__(self, transpose=True, data_format="tchw"):
        super().__init__()
        assert data_format in [
            "tchw",
            "cthw",
        ], f"Target format must in ['tchw', 'cthw'], but got {data_format}"
        self.transpose = transpose
        self.data_format = data_format

    def apply(self, imgs):
        """
        Performs Image to NumpyArray operations.
        Args:
            imgs: List where each item is a PIL.Image.
            For example, [PIL.Image0, PIL.Image1, PIL.Image2, ...]
        return:
            np_imgs: Numpy array.
        """
        t_imgs = np.stack(imgs).astype("float32")
        if self.transpose:
            if self.data_format == "tchw":
                t_imgs = t_imgs.transpose([0, 3, 1, 2])  # tchw
            else:
                t_imgs = t_imgs.transpose([3, 0, 1, 2])  # cthw
        imgs = t_imgs
        return {
            "imgs": imgs,
        }


class NormalizeVideo(BaseComponent):
    """
    Normalization.
    """

    INPUT_KEYS = ["imgs"]
    OUTPUT_KEYS = ["img", "img"]
    DEAULT_INPUTS = {"imgs": "imgs"}
    DEAULT_OUTPUTS = {
        "img": "img",
    }

    def __init__(self, mean, std, tensor_shape=[3, 1, 1], inplace=False):
        super().__init__()

        self.inplace = inplace
        if not inplace:
            self.mean = np.array(mean).reshape(tensor_shape).astype(np.float32)
            self.std = np.array(std).reshape(tensor_shape).astype(np.float32)
        else:
            self.mean = np.array(mean, dtype=np.float32)
            self.std = np.array(std, dtype=np.float32)

    def apply(self, imgs):
        """
        Performs normalization operations.
        Args:
            imgs: Numpy array.
        return:
            np_imgs: Numpy array after normalization.
        """

        if self.inplace:
            n = len(imgs)
            h, w, c = imgs[0].shape
            norm_imgs = np.empty((n, h, w, c), dtype=np.float32)
            for i, img in enumerate(imgs):
                norm_imgs[i] = img

            for img in norm_imgs:  # [n,h,w,c]
                mean = np.float64(self.mean.reshape(1, -1))  # [1, 3]
                stdinv = 1 / np.float64(self.std.reshape(1, -1))  # [1, 3]
                cv2.subtract(img, mean, img)
                cv2.multiply(img, stdinv, img)
        else:
            imgs = imgs
            norm_imgs = imgs / 255.0
            norm_imgs -= self.mean
            norm_imgs /= self.std

        imgs = norm_imgs
        imgs = np.expand_dims(imgs, axis=0).copy()
        return {
            "img": imgs,
        }
