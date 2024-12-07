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
from ....modules.image_classification.model_list import MODELS
from ..base import BasicPredictor
from ..common.vision import *
from .processors import *
from .result import TopkResult


class ClasPredictor(BasicPredictor):

    entities = MODELS

    _FUNC_MAP = {}
    register = FuncRegister(_FUNC_MAP)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.preprocessors, self.predictor, self.postprocessors = self._build()

    def _build_batch_sampler(self):
        return ImageBatchSampler()

    def _get_result_class(self):
        return TopkResult

    def _build(self):
        preprocessors = {"Read": ReadImage(format="RGB")}
        for cfg in self.config["PreProcess"]["transform_ops"]:
            tf_key = list(cfg.keys())[0]
            func = self._FUNC_MAP[tf_key]
            args = cfg.get(tf_key, {})
            name, op = func(self, **args) if args else func(self)
            preprocessors[name] = op

        predictor = ImagePredictor(
            model_dir=self.model_dir,
            model_prefix=self.MODEL_FILE_PREFIX,
            option=self.pp_option,
        )

        postprocessors = {}
        for key in self.config["PostProcess"]:
            func = self._FUNC_MAP.get(key)
            args = self.config["PostProcess"].get(key, {})
            name, op = func(self, **args) if args else func(self)
            postprocessors[name] = op
        return preprocessors, predictor, postprocessors

    def process(self, batch_data):
        batch_raw_imgs = self.preprocessors["Read"](imgs=batch_data)
        batch_imgs = self.preprocessors["Resize"](imgs=batch_raw_imgs)
        batch_imgs = self.preprocessors["Crop"](imgs=batch_imgs)
        batch_imgs = self.preprocessors["Normalize"](imgs=batch_imgs)
        batch_imgs = self.preprocessors["ToCHW"](imgs=batch_imgs)
        batch_preds = self.predictor(imgs=batch_imgs)
        batch_class_ids, batch_scores, batch_label_names = self.postprocessors["Topk"](
            batch_preds
        )
        return {
            "input_path": batch_data,
            "input_img": batch_raw_imgs,
            "class_ids": batch_class_ids,
            "scores": batch_scores,
            "label_names": batch_label_names,
        }

    @register("ResizeImage")
    # TODO(gaotingquan): backend & interpolation
    def build_resize(
        self, resize_short=None, size=None, backend="cv2", interpolation="LINEAR"
    ):
        assert resize_short or size
        if resize_short:
            op = ResizeByShort(
                target_short_edge=resize_short, size_divisor=None, interp="LINEAR"
            )
        else:
            op = Resize(target_size=size)
        return "Resize", op

    @register("CropImage")
    def build_crop(self, size=224):
        return "Crop", Crop(crop_size=size)

    @register("NormalizeImage")
    def build_normalize(
        self,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        scale=1 / 255,
        order="",
        channel_num=3,
    ):
        assert channel_num == 3
        assert order == ""
        return "Normalize", Normalize(scale=scale, mean=mean, std=std)

    @register("ToCHWImage")
    def build_to_chw(self):
        return "ToCHW", ToCHWImage()

    @register("Topk")
    def build_topk(self, topk, label_list=None):
        return "Topk", Topk(topk=int(topk), class_ids=label_list)
