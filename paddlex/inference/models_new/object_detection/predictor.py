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

from typing import Any, List, Sequence, Optional, Union, Tuple

import numpy as np

from ....utils.func_register import FuncRegister
from ....modules.object_detection.model_list import MODELS
from ...common.batch_sampler import ImageBatchSampler

from ..common import StaticInfer
from ..base import BasicPredictor
from .processors import (
    DetPad,
    DetPostProcess,
    Normalize,
    PadStride,
    ReadImage,
    Resize,
    ToBatch,
    ToCHWImage,
    WarpAffine,
)
from .result import DetResult
from .utils import STATIC_SHAPE_MODEL_LIST


class DetPredictor(BasicPredictor):

    entities = MODELS

    _FUNC_MAP = {}
    register = FuncRegister(_FUNC_MAP)

    def __init__(
        self,
        *args,
        imgsz: Optional[Union[int, Tuple[int, int]]] = None,
        threshold: Optional[float] = None,
        **kwargs,
    ):
        """Initializes DetPredictor.
        Args:
            *args: Arbitrary positional arguments passed to the superclass.
            imgsz (Optional[Union[int, Tuple[int, int]]], optional): The input image size (w, h). Defaults to None.
            threshold (Optional[float], optional): The threshold for filtering out low-confidence predictions.
                Defaults to None.
            **kwargs: Arbitrary keyword arguments passed to the superclass.
        """
        super().__init__(*args, **kwargs)

        if imgsz is not None:
            assert (
                self.model_name not in STATIC_SHAPE_MODEL_LIST
            ), f"The model {self.model_name} is not supported set input shape"
            if isinstance(imgsz, int):
                imgsz = (imgsz, imgsz)
            elif isinstance(imgsz, (tuple, list)):
                assert len(imgsz) == 2, f"The length of `imgsz` should be 2."
            else:
                raise ValueError(
                    f"The type of `imgsz` must be int or Tuple[int, int], but got {type(imgsz)}."
                )
        self.imgsz = imgsz
        self.threshold = threshold
        self.pre_ops, self.infer, self.post_op = self._build()

    def _build_batch_sampler(self):
        return ImageBatchSampler()

    def _get_result_class(self):
        return DetResult

    def _build(self) -> Tuple:
        """Build the preprocessors, inference engine, and postprocessors based on the configuration.

        Returns:
            tuple: A tuple containing the preprocessors, inference engine, and postprocessors.
        """
        # build preprocess ops
        pre_ops = [ReadImage(format="RGB")]
        for cfg in self.config["Preprocess"]:
            tf_key = cfg["type"]
            func = self._FUNC_MAP[tf_key]
            cfg.pop("type")
            args = cfg
            op = func(self, **args) if args else func(self)
            if op:
                pre_ops.append(op)
        pre_ops.append(self.build_to_batch())
        if self.imgsz is not None:
            if isinstance(pre_ops[1], Resize):
                pre_ops.pop(1)
            pre_ops.insert(1, self.build_resize(self.imgsz, False, 2))

        # build infer
        infer = StaticInfer(
            model_dir=self.model_dir,
            model_prefix=self.MODEL_FILE_PREFIX,
            option=self.pp_option,
        )

        # build postprocess op
        post_op = self.build_postprocess()

        return pre_ops, infer, post_op

    def _format_output(self, pred: Sequence[Any]) -> List[dict]:
        """
        Transform batch outputs into a list of single image output.

        Args:
            pred (Sequence[Any]): The input predictions, which can be either a list of 3 or 4 elements.
                - When len(pred) == 4, it is expected to be in the format [boxes, class_ids, scores, masks],
                  compatible with SOLOv2 output.
                - When len(pred) == 3, it is expected to be in the format [boxes, box_nums, masks],
                  compatible with Instance Segmentation output.

        Returns:
            List[dict]: A list of dictionaries, each containing either 'class_id' and 'masks' (for SOLOv2),
                or 'boxes' and 'masks' (for Instance Segmentation), or just 'boxes' if no masks are provided.
        """
        box_idx_start = 0
        pred_box = []

        if len(pred) == 4:
            # Adapt to SOLOv2
            pred_class_id = []
            pred_mask = []
            pred_class_id.append([pred[1], pred[2]])
            pred_mask.append(pred[3])
            return [
                {
                    "class_id": np.array(pred_class_id[i]),
                    "masks": np.array(pred_mask[i]),
                }
                for i in range(len(pred_class_id))
            ]

        if len(pred) == 3:
            # Adapt to Instance Segmentation
            pred_mask = []
        for idx in range(len(pred[1])):
            np_boxes_num = pred[1][idx]
            box_idx_end = box_idx_start + np_boxes_num
            np_boxes = pred[0][box_idx_start:box_idx_end]
            pred_box.append(np_boxes)
            if len(pred) == 3:
                np_masks = pred[2][box_idx_start:box_idx_end]
                pred_mask.append(np_masks)
            box_idx_start = box_idx_end

        if len(pred) == 3:
            return [
                {"boxes": np.array(pred_box[i]), "masks": np.array(pred_mask[i])}
                for i in range(len(pred_box))
            ]
        else:
            return [{"boxes": np.array(res)} for res in pred_box]

    def process(self, batch_data: List[Any], threshold: Optional[float] = None):
        """
        Process a batch of data through the preprocessing, inference, and postprocessing.

        Args:
            batch_data (List[Union[str, np.ndarray], ...]): A batch of input data (e.g., image file paths).

        Returns:
            dict: A dictionary containing the input path, raw image, class IDs, scores, and label names
                for every instance of the batch. Keys include 'input_path', 'input_img', 'class_ids', 'scores', and 'label_names'.
        """
        datas = batch_data
        # preprocess
        for pre_op in self.pre_ops[:-1]:
            datas = pre_op(datas)

        # use `ToBatch` format batch inputs
        batch_inputs = self.pre_ops[-1](datas)

        # do infer
        batch_preds = self.infer(batch_inputs)

        # process a batch of predictions into a list of single image result
        preds_list = self._format_output(batch_preds)

        # postprocess
        boxes = self.post_op(
            preds_list, datas, threshold if threshold is not None else self.threshold
        )

        return {
            "input_path": [data.get("img_path", None) for data in datas],
            "input_img": [data["ori_img"] for data in datas],
            "boxes": boxes,
        }

    @register("Resize")
    def build_resize(self, target_size, keep_ratio=False, interp=2):
        assert target_size
        if isinstance(interp, int):
            interp = {
                0: "NEAREST",
                1: "LINEAR",
                2: "BICUBIC",
                3: "AREA",
                4: "LANCZOS4",
            }[interp]
        op = Resize(target_size=target_size[::-1], keep_ratio=keep_ratio, interp=interp)
        return op

    @register("NormalizeImage")
    def build_normalize(
        self,
        norm_type=None,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        is_scale=True,
    ):
        if is_scale:
            scale = 1.0 / 255.0
        else:
            scale = 1
        if not norm_type or norm_type == "none":
            norm_type = "mean_std"
        if norm_type != "mean_std":
            mean = 0
            std = 1
        return Normalize(scale=scale, mean=mean, std=std)

    @register("Permute")
    def build_to_chw(self):
        return ToCHWImage()

    @register("Pad")
    def build_pad(self, fill_value=None, size=None):
        if fill_value is None:
            fill_value = [127.5, 127.5, 127.5]
        if size is None:
            size = [3, 640, 640]
        return DetPad(size=size, fill_value=fill_value)

    @register("PadStride")
    def build_pad_stride(self, stride=32):
        return PadStride(stride=stride)

    @register("WarpAffine")
    def build_warp_affine(self, input_h=512, input_w=512, keep_res=True):
        return WarpAffine(input_h=input_h, input_w=input_w, keep_res=keep_res)

    def build_to_batch(self):
        models_required_imgsize = [
            "DETR",
            "DINO",
            "RCNN",
            "YOLOv3",
            "CenterNet",
            "BlazeFace",
            "BlazeFace-FPN-SSH",
        ]
        if any(name in self.model_name for name in models_required_imgsize):
            ordered_required_keys = (
                "img_size",
                "img",
                "scale_factors",
            )
        else:
            ordered_required_keys = ("img", "scale_factors")

        return ToBatch(ordered_required_keys=ordered_required_keys)

    def build_postprocess(self):
        return DetPostProcess(
            threshold=self.config["draw_threshold"],
            labels=self.config["label_list"],
            layout_postprocess=self.config.get("layout_postprocess", False),
        )
