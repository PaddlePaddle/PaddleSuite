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
import os.path as osp
from typing import List, Sequence, Union, Optional, Tuple

import re
import numpy as np
import cv2
import math
import json
import tempfile
import lazy_paddle


class ResizeVideo:
    """Scale images."""

    def __init__(
        self,
        target_size: int = 224,
    ) -> None:
        """
        Initializes the Scale class.

        Args:
            short_size (int): The target size for the shorter side of the image.
            fixed_ratio (bool): Whether to maintain a fixed aspect ratio of 4:3.
            keep_ratio (Union[bool, None]): Whether to keep the aspect ratio. Cannot be True if fixed_ratio is True.
            do_round (bool): Whether to round the scaling factor.
        """
        super().__init__()
        self.target_size = target_size

    def resize(self, video: List[np.ndarray]) -> List[np.ndarray]:
        """
        Performs resize operations on a sequence of images.

        Args:
            video (List[np.ndarray]): List where each item is an image,  as a numpy array.
             For example, [np.ndarray0, np.ndarray1, np.ndarray2, ...]

        Returns:
            List[np.ndarray]: List where each item is a np.ndarray after scaling.
        """

        num_seg = len(video)
        seg_len = len(video[0])

        for i in range(num_seg):
            for j in range(seg_len):
                img = video[i][j]
                if isinstance(img, np.ndarray):
                    h, w, _ = img.shape
                else:
                    raise NotImplementedError
                video[i][j] = cv2.resize(img, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)
        data = {}
        data['video'] = video
        data['ori_img_size'] = [h, w]
        return data

    def __call__(self, videos: List[np.ndarray]) -> List[np.ndarray]:
        """
        Apply the scaling operation to a list of videos.

        Args:
            videos (List[np.ndarray]): A list of videos, where each video is a sequence
            of images.

        Returns:
            List[np.ndarray]: A list of videos after scaling, where each video is a list of images.
        """
        return [self.resize(video) for video in videos]


class Image2Array:
    """Convert a sequence of images to a numpy array with optional transposition."""

    def __init__(self, transpose: bool = True, data_format: str = "tchw") -> None:
        """
        Initializes the Image2Array class.

        Args:
            transpose (bool): Whether to transpose the resulting numpy array.
            data_format (str): The format to transpose to, either 'tchw' or 'cthw'.

        Raises:
            AssertionError: If data_format is not one of the allowed values.
        """
        super().__init__()
        assert data_format in [
            "tchw",
            "cthw",
        ], f"Target format must in ['tchw', 'cthw'], but got {data_format}"
        self.transpose = transpose
        self.data_format = data_format

    def img2array(self, videos: List[np.ndarray]) -> np.ndarray:
        """
        Converts a sequence of images to a numpy array and optionally transposes it.

        Args:
            imgs (List[np.ndarray]): A list of images to be converted to a numpy array.

        Returns:
            np.ndarray: A numpy array representation of the images.
        """
        video_to_array = videos['video']
        num_seg = len(video_to_array)
        for i in range(num_seg):
            video_one = video_to_array[i]
            video_one = [img.transpose([2, 0, 1]) for img in video_one]
            video_one = np.concatenate(
                [np.expand_dims(img, axis=1) for img in video_one], axis=1
            )
            video_to_array[i] = video_one
        videos['video'] = video_to_array

        return videos

    def __call__(self, videos: List[np.ndarray]) -> List[np.ndarray]:
        """
        Apply the image to array conversion to a list of videos.

        Args:
            videos (List[Sequence[np.ndarray]]): A list of videos, where each video is a sequence of images.

        Returns:
            List[np.ndarray]: A list of numpy arrays, one for each video.
        """
        return [self.img2array(video) for video in videos]


class NormalizeVideo:
    """
    Normalize video frames by subtracting the mean and dividing by the standard deviation.
    """

    def __init__(
        self,
        scale: float = 255.,
    ) -> None:
        """
        Initializes the NormalizeVideo class.

        Args:
            mean (Sequence[float]): The mean values for each channel.
            std (Sequence[float]): The standard deviation values for each channel.
            tensor_shape (Sequence[int]): The shape of the mean and std tensors.
            inplace (bool): Whether to perform normalization in place.
        """
        super().__init__()

        self.scale = scale

    def normalize_video(self, videos: np.ndarray) -> np.ndarray:
        """
        Normalizes a sequence of images.

        Args:
            imgs (np.ndarray): A numpy array of images to be normalized.

        Returns:
            np.ndarray: The normalized images as a numpy array.
        """
        video_to_array = videos['video']
        num_seg = len(video_to_array)
        for i in range(num_seg):
            video_to_array[i]  = video_to_array[i].astype(np.float32) / self.scale
            video_to_array[i] = np.expand_dims(video_to_array[i], axis=0)

        return video_to_array

    def __call__(self, videos: List[np.ndarray]) -> List[np.ndarray]:
        """
        Apply normalization to a list of videos.

        Args:
            videos (List[np.ndarray]): A list of videos, where each video is a numpy array of images.

        Returns:
            List[np.ndarray]: A list of normalized videos as numpy arrays.
        """
        return [self.normalize_video(video) for video in videos]


