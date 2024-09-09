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
from functools import partial, wraps

from ...modules.text_detection.model_list import MODELS

from ..components import *
from .base import BasePredictor


def register(register_map, key):
    """register the option setting func"""

    def decorator(func):
        register_map[key] = func

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            return func(self, *args, **kwargs)

        return wrapper

    return decorator


class TextDetPredictor(BasePredictor):

    entities = MODELS

    INPUT_KEYS = "x"
    OUTPUT_KEYS = "text_det_res"
    DEAULT_INPUTS = {"x": "x"}
    DEAULT_OUTPUTS = {"text_det_res": "text_det_res"}

    _REGISTER_MAP = {}
    register2self = partial(register, _REGISTER_MAP)

    def _build_components(self):
        ops = {}
        for cfg in self.config["PreProcess"]["transform_ops"]:
            tf_key = list(cfg.keys())[0]
            func = self._REGISTER_MAP.get(tf_key)
            args = cfg.get(tf_key, {})
            op = func(self, **args) if args else func(self)
            if op:
                ops[tf_key] = op

        kernel_option = PaddlePredictorOption()
        # kernel_option.set_device(self.device)
        predictor = ImagePredictor(
            model_dir=self.model_dir,
            model_prefix=self.MODEL_FILE_PREFIX,
            option=kernel_option,
        )
        predictor.set_inputs({"imgs": "img"})
        ops["predictor"] = predictor

        key, op = self.build_postprocess(**self.config["PostProcess"])
        ops[key] = op
        return ops

    @register2self("DecodeImage")
    def build_readimg(self, channel_first, img_mode):
        assert channel_first == False
        return ReadImage(format=img_mode, batch_size=self.kwargs.get("batch_size", 1))

    @register2self("DetResizeForTest")
    def build_resize(self, resize_long=960):
        return DetResizeForTest(limit_side_len=resize_long, limit_type="max")

    @register2self("NormalizeImage")
    def build_normalize(
        self,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        scale=1 / 255,
        order="",
        channel_num=3,
    ):
        return NormalizeImage(
            mean=mean, std=std, scale=scale, order=order, channel_num=channel_num
        )

    @register2self("ToCHWImage")
    def build_to_chw(self):
        return ToCHWImage()

    def build_postprocess(self, **kwargs):
        if kwargs.get("name") == "DBPostProcess":
            return "DBPostProcess", DBPostProcess(
                thresh=kwargs.get("thresh", 0.3),
                box_thresh=kwargs.get("box_thresh", 0.7),
                max_candidates=kwargs.get("max_candidates", 1000),
                unclip_ratio=kwargs.get("unclip_ratio", 2.0),
                use_dilation=kwargs.get("use_dilation", False),
                score_mode=kwargs.get("score_mode", "fast"),
                box_type=kwargs.get("box_type", "quad"),
            )

        else:
            raise Exception()

    @register2self("DetLabelEncode")
    def foo(self, *args, **kwargs):
        return None

    @register2self("KeepKeys")
    def foo(self, *args, **kwargs):
        return None
