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
from .pp_shitu_v2 import ShiTuV2Pipeline
from ..results import FaceRecResult


class FaceRecPipeline(ShiTuV2Pipeline):
    """Face Recognition Pipeline"""

    entities = "face_recognition"

    def get_rec_result(self, det_res, indexer):
        if len(det_res["boxes"]) == 0:
            return {"label": [], "score": []}
        subs_of_img = list(self._crop_by_boxes(det_res))
        img_list = [img["img"] for img in subs_of_img]
        all_rec_res = list(self.rec_model(img_list))
        all_rec_res = next(indexer(all_rec_res))
        output = {"label": [], "score": []}
        for res in all_rec_res:
            output["label"].append(res["label"])
            output["score"].append(res["score"])
        return output

    def get_final_result(self, det_res, rec_res):
        single_img_res = {"input_path": det_res["input_path"], "boxes": []}
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
        return FaceRecResult(single_img_res)
