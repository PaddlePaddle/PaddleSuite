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

PIPELINE_ARGUMENTS = {
    "OCR": [
        {
            "name": "--use_doc_orientation_classify",
            "type": bool,
            "help": "Determines whether to use document orientation classification",
        },
        {
            "name": "--use_doc_unwarping",
            "type": bool,
            "help": "Determines whether to use document unwarping",
        },
        {
            "name": "--use_textline_orientation",
            "type": bool,
            "help": "Determines whether to consider text line orientation",
        },
        {
            "name": "--text_det_limit_side_len",
            "type": int,
            "help": "Sets the side length limit for text detection.",
        },
        {
            "name": "--text_det_limit_type",
            "type": str,
            "help": "Sets the limit type for text detection.",
        },
        {
            "name": "--text_det_thresh",
            "type": float,
            "help": "Sets the threshold for text detection.",
        },
        {
            "name": "--text_det_box_thresh",
            "type": float,
            "help": "Sets the box threshold for text detection.",
        },
        {
            "name": "--text_det_max_candidates",
            "type": int,
            "help": "Sets the maximum number of candidate boxes for text detection.",
        },
        {
            "name": "--text_det_unclip_ratio",
            "type": float,
            "help": "Sets the unclip ratio for text detection.",
        },
        {
            "name": "--text_det_use_dilation",
            "type": bool,
            "help": "Determines whether to use dilation in text detection.",
        },
        {
            "name": "--text_rec_score_thresh",
            "type": float,
            "help": "Sets the score threshold for text recognition.",
        },
    ],
    "object_detection": [
        {
            "name": "--threshold",
            "type": float,
            "help": "Sets the threshold for object detection.",
        },
    ],
    "image_classification": [
        {
            "name": "--topk",
            "type": int,
            "help": "Sets the Top-K value for image classification.",
        },
    ],
    "image_multilabel_classification": [
        {
            "name": "--threshold",
            "type": float,
            "help": "Sets the threshold for image multilabel classification.",
        },
    ],
    "pedestrian_attribute_recognition": [
        {
            "name": "--det_threshold",
            "type": float,
            "help": "Sets the threshold for human detection.",
        },
        {
            "name": "--cls_threshold",
            "type": float,
            "help": "Sets the threshold for pedestrian attribute recognition.",
        },
    ],
    "vehicle_attribute_recognition": [
        {
            "name": "--det_threshold",
            "type": float,
            "help": "Sets the threshold for vehicle detection.",
        },
        {
            "name": "--cls_threshold",
            "type": float,
            "help": "Sets the threshold for vehicle attribute recognition.",
        },
    ],
    "ts_classification": None,
    "formula_recognition": None,
    "video_classification": [
        {
            "name": "--topk",
            "type": int,
            "help": "Sets the Top-K value for video classification.",
        },
    ],
}
