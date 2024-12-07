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

from ...utils.func_register import FuncRegister
from ...modules.video_classification.model_list import MODELS
from ..components import *
from ..results import TopkVideoResult
from .base import BasicPredictor


class VideoClasPredictor(BasicPredictor):

    entities = [*MODELS]

    _FUNC_MAP = {}
    register = FuncRegister(_FUNC_MAP)

    def _build_components(self):

        for cfg in self.config["PreProcess"]["transform_ops"]:
            tf_key = list(cfg.keys())[0]
            func = self._FUNC_MAP[tf_key]
            args = cfg.get(tf_key, {})
            op = func(self, **args) if args else func(self)
            self._add_component(op)

        predictor = VideoPredictor(
            model_dir=self.model_dir,
            model_prefix=self.MODEL_FILE_PREFIX,
            option=self.pp_option,
        )
        self._add_component(predictor)

        post_processes = self.config["PostProcess"]
        for key in post_processes:
            func = self._FUNC_MAP.get(key)
            args = post_processes.get(key, {})
            op = func(self, **args) if args else func(self)
            self._add_component(op)

    @register("ReadVideo")
    def build_readvideo(
        self,
        num_seg=8,
        target_size=224,
        seg_len=1,
        sample_type=None,
    ):
        op = ReadVideo(
            backend="decord",
            num_seg=num_seg,
            seg_len=seg_len,
            sample_type=sample_type,
        )
        return op

    @register("Scale")
    def build_scale(self, short_size=224):
        return Scale(
            short_size=short_size,
            fixed_ratio=True,
            keep_ratio=None,
            do_round=False,
        )

    @register("CenterCrop")
    def build_center_crop(self, target_size=224):
        return CenterCrop(target_size=target_size)

    @register("Image2Array")
    def build_image2array(self, data_format="tchw"):
        return Image2Array(transpose=True, data_format="tchw")

    @register("NormalizeVideo")
    def build_normalize(
        self,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ):
        return NormalizeVideo(mean=mean, std=std)

    @register("Topk")
    def build_topk(self, topk, label_list=None):
        return VideoClasTopk(topk=int(topk), class_ids=label_list)

    def _pack_res(self, single):
        keys = ["input_path", "class_ids", "scores"]
        if "label_names" in single:
            keys.append("label_names")
        return TopkVideoResult({key: single[key] for key in keys})
