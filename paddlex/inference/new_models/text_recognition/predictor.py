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
from ..common.vision import *
from .processors import *
from .result import TextRecResult


class TextRecPredictor(BasicPredictor):

    entities = MODELS

    _FUNC_MAP = {}
    register = FuncRegister(_FUNC_MAP)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pre_tfs, self.predictor, self.post_op = self._build()

    def _build_batch_sampler(self):
        return ImageBatchSampler()

    def _get_result_class(self):
        return TextRecResult

    def _build(self):
        pre_tfs = {"ReadImage": ReadImage(format="RGB")}
        for cfg in self.config["PreProcess"]["transform_ops"]:
            tf_key = list(cfg.keys())[0]
            assert tf_key in self._FUNC_MAP
            func = self._FUNC_MAP[tf_key]
            args = cfg.get(tf_key, {})
            op = func(self, **args) if args else func(self)
            if op:
                pre_tfs[op.name] = op

        predictor = ImagePredictor(
            model_dir=self.model_dir,
            model_prefix=self.MODEL_FILE_PREFIX,
            option=self.pp_option,
        )

        post_op = self.build_postprocess(**self.config["PostProcess"])
        return pre_tfs, predictor, post_op

    def process(self, batch_data):
        batch_raw_imgs = self.pre_tfs["ReadImage"](imgs=batch_data)
        batch_imgs = self.pre_tfs["OCRReisizeNormImg"](imgs=batch_raw_imgs)
        batch_preds = self.predictor(imgs=batch_imgs)
        texts, scores = self.post_op(batch_preds)
        return {
            "input_path": batch_data,
            "input_img": batch_raw_imgs,
            "rec_text": texts,
            "rec_score": scores,
        }

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