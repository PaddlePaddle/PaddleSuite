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

import numpy as np

from ....utils import logging
from ..base import BaseStaticInfer

__all__ = ["Topk", "ImagePredictor"]


class ImagePredictor(BaseStaticInfer):

    def to_batch(self, imgs):
        return [np.stack(imgs, axis=0).astype(dtype=np.float32, copy=False)]

    def format_output(self, pred):
        return pred[0]


class Topk:
    """Topk Transform"""

    def __init__(self, topk, class_ids=None):
        super().__init__()
        assert isinstance(topk, (int,))
        self.topk = topk
        self.class_id_map = self._parse_class_id_map(class_ids)

    def _parse_class_id_map(self, class_ids):
        """parse class id to label map file"""
        if class_ids is None:
            return None
        class_id_map = {id: str(lb) for id, lb in enumerate(class_ids)}
        return class_id_map

    def __call__(self, preds):
        indexes = preds.argsort(axis=1)[:, -self.topk :][:, ::-1].astype("int32")
        scores = [
            np.around(pred[index], decimals=5) for pred, index in zip(preds, indexes)
        ]
        label_names = [[self.class_id_map[i] for i in index] for index in indexes]
        return indexes, scores, label_names
