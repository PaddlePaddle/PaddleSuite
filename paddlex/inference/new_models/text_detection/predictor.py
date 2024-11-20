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

from ....utils.func_register import FuncRegister
from ....modules.text_detection.model_list import MODELS
from ..base import BasicPredictor
from ..common.cv_components import *
from ..common.paddle_predictor import ImagePredictor
from .transformers import *
from .result import TextDetResult


class TextDetPredictor(BasicPredictor):

    entities = MODELS

    _FUNC_MAP = {}
    register = FuncRegister(_FUNC_MAP)

    def _build_batch_sampler(self):
        return ImageBatchSampler()

    def _get_result_class(self):
        return TextDetResult

    def _build_transformers(self):
        for cfg in self.config["PreProcess"]["transform_ops"]:
            tf_key = list(cfg.keys())[0]
            func = self._FUNC_MAP[tf_key]
            args = cfg.get(tf_key, {})
            op = func(self, **args) if args else func(self)
            if op:
                self._add_transformer(op)

        predictor = ImagePredictor(
            model_dir=self.model_dir,
            model_prefix=self.MODEL_FILE_PREFIX,
            option=self.pp_option,
        )
        self._add_transformer(predictor)

        op = self.build_postprocess(**self.config["PostProcess"])
        self._add_transformer(op)

    def _set_dataflow(self):
        self.ReadImage.inputs.img.fetch(self.batch_sampler.outputs.img)
        self.DetResizeForTest.inputs.img.fetch(self.ReadImage.outputs.img)
        self.NormalizeImage.inputs.img.fetch(self.DetResizeForTest.outputs.img)
        self.ToCHWImage.inputs.img.fetch(self.NormalizeImage.outputs.img)
        self.ImagePredictor.inputs.img.fetch(self.ToCHWImage.outputs.img)
        self.DBPostProcess.inputs.pred.fetch(self.ImagePredictor.outputs.pred)
        self.DBPostProcess.inputs.img_shape.fetch(
            self.DetResizeForTest.outputs.img_shape
        )
        self.result_packager.inputs.input_img.fetch(self.ReadImage.outputs.img)
        self.result_packager.inputs.input_path.fetch(self.batch_sampler.outputs.img)
        self.result_packager.inputs.polys.fetch(self.DBPostProcess.outputs.polys)
        self.result_packager.inputs.scores.fetch(self.DBPostProcess.outputs.scores)

    @register("DecodeImage")
    def build_readimg(self, channel_first, img_mode):
        assert channel_first == False
        return ReadImage(format=img_mode)

    @register("DetResizeForTest")
    def build_resize(self, **kwargs):
        # TODO: align to PaddleOCR
        if self.model_name in ("PP-OCRv4_server_det", "PP-OCRv4_mobile_det"):
            resize_long = kwargs.get("resize_long", 960)
            return DetResizeForTest(limit_side_len=resize_long, limit_type="max")
        return DetResizeForTest(**kwargs)

    @register("NormalizeImage")
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

    @register("ToCHWImage")
    def build_to_chw(self):
        return ToCHWImage()

    def build_postprocess(self, **kwargs):
        if kwargs.get("name") == "DBPostProcess":
            return DBPostProcess(
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

    @register("DetLabelEncode")
    def foo(self, *args, **kwargs):
        return None

    @register("KeepKeys")
    def foo(self, *args, **kwargs):
        return None
