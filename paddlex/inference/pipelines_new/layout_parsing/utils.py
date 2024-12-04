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

__all__ = ["convert_points_to_boxes", "get_sub_regions_ocr_res"]

import numpy as np
import copy
from ..ocr.result import OCRResult


def convert_points_to_boxes(dt_polys: list) -> np.ndarray:
    """
    Converts a list of polygons to a numpy array of bounding boxes.

    Args:
        dt_polys (list): A list of polygons, where each polygon is represented
                        as a list of (x, y) points.

    Returns:
        np.ndarray: A numpy array of bounding boxes, where each box is represented
                    as [left, top, right, bottom].
                    If the input list is empty, returns an empty numpy array.
    """

    if len(dt_polys) > 0:
        dt_polys_tmp = dt_polys.copy()
        dt_polys_tmp = np.array(dt_polys_tmp)
        boxes_left = np.min(dt_polys_tmp[:, :, 0], axis=1)
        boxes_right = np.max(dt_polys_tmp[:, :, 0], axis=1)
        boxes_top = np.min(dt_polys_tmp[:, :, 1], axis=1)
        boxes_bottom = np.max(dt_polys_tmp[:, :, 1], axis=1)
        dt_boxes = np.array([boxes_left, boxes_top, boxes_right, boxes_bottom])
        dt_boxes = dt_boxes.T
    else:
        dt_boxes = np.array([])
    return dt_boxes


def get_overlap_boxes_idx(src_boxes: np.ndarray, ref_boxes: np.ndarray) -> list:
    """
    Get the indices of source boxes that overlap with reference boxes based on a specified threshold.

    Args:
        src_boxes (np.ndarray): A 2D numpy array of source bounding boxes.
        ref_boxes (np.ndarray): A 2D numpy array of reference bounding boxes.

    Returns:
        list: A list of indices of source boxes that overlap with any reference box.
    """
    match_idx_list = []
    src_boxes_num = len(src_boxes)
    if src_boxes_num > 0 and len(ref_boxes) > 0:
        for rno in range(len(ref_boxes)):
            ref_box = ref_boxes[rno]
            x1 = np.maximum(ref_box[0], src_boxes[:, 0])
            y1 = np.maximum(ref_box[1], src_boxes[:, 1])
            x2 = np.minimum(ref_box[2], src_boxes[:, 2])
            y2 = np.minimum(ref_box[3], src_boxes[:, 3])
            pub_w = x2 - x1
            pub_h = y2 - y1
            match_idx = np.where((pub_w > 3) & (pub_h > 3))[0]
            match_idx_list.extend(match_idx)
    return match_idx_list


def get_sub_regions_ocr_res(
    overall_ocr_res: OCRResult, object_boxes: list, flag_within: bool = True
) -> OCRResult:
    """
    Filters OCR results to only include text boxes within specified object boxes based on a flag.

    Args:
        overall_ocr_res (OCRResult): The original OCR result containing all text boxes.
        object_boxes (list): A list of bounding boxes for the objects of interest.
        flag_within (bool): If True, only include text boxes within the object boxes. If False, exclude text boxes within the object boxes.

    Returns:
        OCRResult: A filtered OCR result containing only the relevant text boxes.
    """
    sub_regions_ocr_res = copy.deepcopy(overall_ocr_res)
    sub_regions_ocr_res["input_img"] = overall_ocr_res["input_img"]
    sub_regions_ocr_res["img_id"] = -1
    sub_regions_ocr_res["dt_polys"] = []
    sub_regions_ocr_res["rec_text"] = []
    sub_regions_ocr_res["rec_score"] = []
    sub_regions_ocr_res["dt_boxes"] = []

    overall_text_boxes = overall_ocr_res["dt_boxes"]
    match_idx_list = get_overlap_boxes_idx(overall_text_boxes, object_boxes)
    match_idx_list = list(set(match_idx_list))
    for box_no in range(len(overall_text_boxes)):
        if flag_within:
            if box_no in match_idx_list:
                flag_match = True
            else:
                flag_match = False
        else:
            if box_no not in match_idx_list:
                flag_match = True
            else:
                flag_match = False
        if flag_match:
            sub_regions_ocr_res["dt_polys"].append(overall_ocr_res["dt_polys"][box_no])
            sub_regions_ocr_res["rec_text"].append(overall_ocr_res["rec_text"][box_no])
            sub_regions_ocr_res["rec_score"].append(
                overall_ocr_res["rec_score"][box_no]
            )
            sub_regions_ocr_res["dt_boxes"].append(overall_ocr_res["dt_boxes"][box_no])
    return sub_regions_ocr_res
