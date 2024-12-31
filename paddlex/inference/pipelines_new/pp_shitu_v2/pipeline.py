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

from typing import Any, Dict, Optional

import pickle
from pathlib import Path
import numpy as np

from ...utils.pp_option import PaddlePredictorOption
from ...common.reader import ReadImage
from ...common.batch_sampler import ImageBatchSampler
from ..components import CropByBoxes, FaissIndexer, FaissBuilder
from ..base import BasePipeline
from .result import ShiTuResult


class ShiTuV2Pipeline(BasePipeline):
    """ShiTuV2 Pipeline"""

    entities = "PP-ShiTuV2"

    def __init__(
        self,
        config: Dict,
        device: str = None,
        pp_option: PaddlePredictorOption = None,
        use_hpip: bool = False,
        hpi_params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            device=device, pp_option=pp_option, use_hpip=use_hpip, hpi_params=hpi_params
        )

        self.det_model = self.create_model(config["SubModules"]["Detection"])
        self.rec_model = self.create_model(config["SubModules"]["Recognition"])
        self._crop_by_boxes = CropByBoxes()
        self._img_reader = ReadImage(format="BGR")

        self._return_k, self._score_thres, self._hamming_radius = (
            config["return_k"],
            config.get("score_thres", None),
            config.get("hamming_radius", None),
        )
        index = config.get("index", None)
        self._indexer = self._build_indexer(index=index) if index else None

        self.batch_sampler = ImageBatchSampler(
            batch_size=self.det_model.batch_sampler.batch_size
        )
        self.img_reader = ReadImage(format="BGR")

    def _build_indexer(self, index):
        return FaissIndexer(
            index=index,
            return_k=self._return_k,
            score_thres=self._score_thres,
            hamming_radius=self._hamming_radius,
        )

    def predict(self, input, index=None, **kwargs):
        indexer = self._build_indexer(index) if index is not None else self._indexer
        assert indexer
        for img_id, batch_data in enumerate(self.batch_sampler(input)):
            raw_imgs = self.img_reader(batch_data)
            all_det_res = list(self.det_model(raw_imgs))
            for input_data, raw_img, det_res in zip(batch_data, raw_imgs, all_det_res):
                rec_res = self.get_rec_result(raw_img, det_res, indexer)
                yield self.get_final_result(input_data, raw_img, det_res, rec_res)

    def get_rec_result(self, raw_img, det_res, indexer):
        if len(det_res["boxes"]) == 0:
            w, h = raw_img.shape[:2]
            det_res["boxes"].append(
                {
                    "cls_id": 0,
                    "label": "full_img",
                    "score": 0,
                    "coordinate": [0, 0, h, w],
                }
            )
        subs_of_img = list(self._crop_by_boxes(raw_img, det_res["boxes"]))
        img_list = [img["img"] for img in subs_of_img]
        all_rec_res = list(self.rec_model(img_list))
        all_rec_res = indexer([rec_res["feature"] for rec_res in all_rec_res])
        output = {"label": [], "score": []}
        for res in all_rec_res:
            output["label"].append(res["label"])
            output["score"].append(res["score"])
        return output

    def get_final_result(self, input_data, raw_img, det_res, rec_res):
        single_img_res = {"input_path": input_data, "input_img": raw_img, "boxes": []}
        for i, obj in enumerate(det_res["boxes"]):
            rec_scores = rec_res["score"][i]
            labels = rec_res["label"][i]
            single_img_res["boxes"].append(
                {
                    "labels": labels,
                    "rec_scores": rec_scores,
                    "det_score": obj["score"],
                    "coordinate": obj["coordinate"],
                }
            )
        return ShiTuResult(single_img_res)

    def build_index(
        self,
        gallery_imgs,
        gallery_label,
        metric_type="IP",
        index_type="HNSW32",
        **kwargs
    ):
        return FaissBuilder.build(
            gallery_imgs,
            gallery_label,
            self.rec_model.predict,
            metric_type=metric_type,
            index_type=index_type,
        )

    def remove_index(self, remove_ids, index):
        return FaissBuilder.remove(remove_ids, index)

    def append_index(
        self,
        gallery_imgs,
        gallery_label,
        index,
    ):
        return FaissBuilder.append(
            gallery_imgs,
            gallery_label,
            self.rec_model.predict,
            index,
        )
