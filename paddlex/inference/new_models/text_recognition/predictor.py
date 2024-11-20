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
from ....modules.text_recognition.model_list import MODELS
from ..base import BasicPredictor
from ..common.cv_components import *
from ..common.paddle_predictor import ImagePredictor
from .transformers import *
from .result import TextRecResult


class TextRecPredictor(BasicPredictor):

    entities = MODELS

    _FUNC_MAP = {}
    register = FuncRegister(_FUNC_MAP)

    def _build_batch_sampler(self):
        return ImageBatchSampler()

    def _get_result_class(self):
        return TextRecResult

    def _build_transformers(self):
        for cfg in self.config["PreProcess"]["transform_ops"]:
            tf_key = list(cfg.keys())[0]
            assert tf_key in self._FUNC_MAP
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
        self.OCRReisizeNormImg.inputs.img.fetch(self.ReadImage.outputs.img)
        self.OCRReisizeNormImg.inputs.img_size.fetch(self.ReadImage.outputs.img_size)
        self.ImagePredictor.inputs.img.fetch(self.OCRReisizeNormImg.outputs.img)
        self.CTCLabelDecode.inputs.pred.fetch(self.ImagePredictor.outputs.pred)
        self.result_packager.inputs.input_path.fetch(self.batch_sampler.outputs.img)
        self.result_packager.inputs.text.fetch(self.CTCLabelDecode.outputs.text)
        self.result_packager.inputs.score.fetch(self.CTCLabelDecode.outputs.score)

    @register("DecodeImage")
    def build_readimg(self, channel_first, img_mode):
        assert channel_first == False
        return ReadImage(format=img_mode)

    @register("RecResizeImg")
    def build_resize(self, image_shape):
        return OCRReisizeNormImg(rec_image_shape=image_shape)

    def build_postprocess(self, **kwargs):
        if kwargs.get("name") == "CTCLabelDecode":
            return CTCLabelDecode(
                character_list=kwargs.get("character_dict"),
            )
        else:
            raise Exception()

    @register("MultiLabelEncode")
    def foo(self, *args, **kwargs):
        return None

    @register("KeepKeys")
    def foo(self, *args, **kwargs):
        return None
