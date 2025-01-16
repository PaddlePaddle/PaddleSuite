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

__all__ = ["convert_points_to_boxes", "get_sub_regions_ocr_res","get_sub_regions_ocr_res","calculate_metrics_with_page"]

import numpy as np
import copy
import json
from typing import List
from nltk.translate.bleu_score import sentence_bleu
from scipy.stats import kendalltau
from ..ocr.result import OCRResult
from ...models_new.object_detection.result import DetResult


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
    sub_regions_ocr_res["doc_preprocessor_image"] = overall_ocr_res[
        "doc_preprocessor_image"
    ]
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
            sub_regions_ocr_res["rec_text"].append(overall_ocr_res["rec_texts"][box_no])
            sub_regions_ocr_res["rec_score"].append(
                overall_ocr_res["rec_scores"][box_no]
            )
            sub_regions_ocr_res["dt_boxes"].append(overall_ocr_res["dt_boxes"][box_no])
    return sub_regions_ocr_res

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1, box2: Lists or tuples representing bounding boxes [x_min, y_min, x_max, y_max].

    Returns:
        float: The IoU value.
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0  # No intersection

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    min_area = min(box1_area, box2_area)

    iou = inter_area / min_area
    return iou

def is_overlaps_y_exceeds_threshold(bbox1, bbox2, overlap_ratio_threshold=0.6):
        _, y0_1, _, y1_1 = bbox1
        _, y0_2, _, y1_2 = bbox2

        overlap = max(0, min(y1_1, y1_2) - max(y0_1, y0_2))
        min_height = min(y1_1 - y0_1, y1_2 - y0_2)

        return (overlap / min_height) > overlap_ratio_threshold

def sort_boxes_from_left_to_right_then_top_to_bottom(layout_bbox, ocr_res, line_height_threshold=0.7):    
    assert ocr_res['boxes'] and ocr_res['rec_texts']

    # span->line->block
    boxes = ocr_res['boxes']
    rec_text = ocr_res['rec_texts']
    x_min,x_max = layout_bbox[0],layout_bbox[2]

    spans = list(zip(boxes, rec_text))
    spans.sort(key=lambda span: span[0][1])
    spans = [list(span) for span in spans]

    lines = []
    first_span = spans[0]
    current_line = [first_span]
    current_y0, current_y1 = first_span[0][1], first_span[0][3]

    for span in spans[1:]:
        y0, y1 = span[0][1],span[0][3]
        if is_overlaps_y_exceeds_threshold((0, current_y0, 0, current_y1), (0, y0, 0, y1), line_height_threshold):
            current_line.append(span)
            current_y0 = min(current_y0, y0)
            current_y1 = max(current_y1, y1)
        else:
            lines.append(current_line)
            current_line = [span]
            current_y0, current_y1 = y0, y1

    if current_line:
        lines.append(current_line)

    for line in lines:
        line.sort(key=lambda span: span[0][0])
        first_span = line[0]
        end_span = line[-1]
        if first_span[0][0] - x_min > 20:
            first_span[1] = '\n ' + first_span[1]
        if x_max - end_span[0][2] > 20:
            end_span[1] = end_span[1]+ '\n'

    ocr_res['boxes'] = [span[0] for line in lines for span in line]
    ocr_res['rec_texts'] = [span[1]+' ' for line in lines for span in line]
    return ocr_res

def get_structure_res(
    overall_ocr_res: OCRResult, layout_det_res: DetResult, table_res_list
) -> OCRResult:
    """
    Extract structured information from OCR and layout detection results.

    Args:
        overall_ocr_res (OCRResult): An object containing the overall OCR results, including detected text boxes and recognized text. The structure is expected to have:
            - "input_img": The image on which OCR was performed.
            - "dt_boxes": A list of detected text box coordinates.
            - "rec_texts": A list of recognized text corresponding to the detected boxes.
            
        layout_det_res (DetResult): An object containing the layout detection results, including detected layout boxes and their labels. The structure is expected to have:
            - "boxes": A list of dictionaries with keys "coordinate" for box coordinates and "label" for the type of content.
            
        table_res_list (list): A list of table detection results, where each item is a dictionary containing:
            - "layout_bbox": The bounding box of the table layout.
            - "pred_html": The predicted HTML representation of the table.

    Returns:
        list: A list of structured boxes where each item is a dictionary containing:
            - "label": The label of the content (e.g., 'table', 'chart', 'image').
            - The label as a key with either table HTML or image data and text.
            - "layout_bbox": The coordinates of the layout box.
    """

    structure_boxes = []
    input_img = overall_ocr_res["input_img"]

    for box_info in layout_det_res["boxes"]:
        layout_bbox = box_info["coordinate"]
        label = box_info["label"]
        rec_res = {'boxes':[],'rec_texts':[],'flag':False}
        drop_index = []
        seg_start_flag = True
        seg_end_flag = True

        if label == 'table':
            for i, table_res in enumerate(table_res_list):
                if calculate_iou(layout_bbox, table_res['layout_bbox']) > 0.5:
                    structure_boxes.append({
                        "label": label,
                        f"{label}": table_res['pred_html'],
                        "layout_bbox": layout_bbox,
                        "seg_start_flag": seg_start_flag,
                        "seg_end_flag": seg_end_flag
                    })
                    del table_res_list[i]
                    break
        else:
            for box_no in range(len(overall_ocr_res["dt_boxes"])):
                overall_text_boxes = overall_ocr_res["dt_boxes"]
                if calculate_iou(layout_bbox, overall_text_boxes[box_no]) > 0.5:
                    rec_res['boxes'].append(overall_text_boxes[box_no])
                    rec_res['rec_texts'].append(overall_ocr_res["rec_texts"][box_no])
                    rec_res['flag'] = True
                    drop_index.append(box_no)
            
            if rec_res['flag']:
                rec_res = sort_boxes_from_left_to_right_then_top_to_bottom(layout_bbox,rec_res,0.7)
                rec_res_first_bbox = rec_res['boxes'][0]
                rec_res_end_bbox = rec_res['boxes'][-1]
                if rec_res_first_bbox[0] - layout_bbox[0] < 10:
                    seg_start_flag = False
                if layout_bbox[2] - rec_res_end_bbox[2] < 10:
                    seg_end_flag = False 

            if label in ['chart', 'image']:
                structure_boxes.append({
                    "label": label,
                    f"{label}": {
                        "img": input_img[int(layout_bbox[1]):int(layout_bbox[3]), int(layout_bbox[0]):int(layout_bbox[2])],
                        # "image_text": ''.join(rec_texts)  # Uncomment if image text is needed
                    },
                    "layout_bbox": layout_bbox,
                    "seg_start_flag": seg_start_flag,
                    "seg_end_flag": seg_end_flag
                })
            else:
                structure_boxes.append({
                    "label": label,
                    f"{label}": ''.join(rec_res['rec_texts']),
                    "layout_bbox": layout_bbox,
                    "seg_start_flag": seg_start_flag,
                    "seg_end_flag": seg_end_flag
                })

    return structure_boxes


def projection_by_bboxes(boxes: np.ndarray, axis: int) -> np.ndarray:
    """
    Generate a 1D projection histogram from bounding boxes along a specified axis.

    Args:
        boxes: A (N, 4) array of bounding boxes defined by [x_min, y_min, x_max, y_max].
        axis: Axis for projection; 0 for horizontal (x-axis), 1 for vertical (y-axis).

    Returns:
        A 1D numpy array representing the projection histogram based on bounding box intervals.
    """
    assert axis in [0, 1]
    max_length = np.max(boxes[:, axis::2])
    projection = np.zeros(max_length, dtype=int)

    # Increment projection histogram over the interval defined by each bounding box
    for start, end in boxes[:, axis::2]:
        projection[start:end] += 1

    return projection

def split_projection_profile(arr_values: np.ndarray, min_value: float, min_gap: float):
    """
    Split the projection profile into segments based on specified thresholds.

    Args:
        arr_values: 1D array representing the projection profile.
        min_value: Minimum value threshold to consider a profile segment significant.
        min_gap: Minimum gap width to consider a separation between segments.

    Returns:
        A tuple of start and end indices for each segment that meets the criteria.
    """
    # Identify indices where the projection exceeds the minimum value
    significant_indices = np.where(arr_values > min_value)[0]
    if not len(significant_indices):
        return

    # Calculate gaps between significant indices
    index_diffs = significant_indices[1:] - significant_indices[:-1]
    gap_indices = np.where(index_diffs > min_gap)[0]

    # Determine start and end indices of segments
    segment_starts = np.insert(significant_indices[gap_indices + 1], 0, significant_indices[0])
    segment_ends = np.append(significant_indices[gap_indices], significant_indices[-1] + 1)

    return segment_starts, segment_ends

def recursive_yx_cut(boxes: np.ndarray, indices: List[int], res: List[int],min_gap = 1):
    """
    Recursively project and segment bounding boxes, starting with Y-axis and followed by X-axis.

    Args:
        boxes: A (N, 4) array representing bounding boxes.
        indices: List of indices indicating the original position of boxes.
        res: List to store indices of the final segmented bounding boxes.
    """
    assert len(boxes) == len(indices)

    # Sort by y_min for Y-axis projection
    y_sorted_indices = boxes[:, 1].argsort()
    y_sorted_boxes = boxes[y_sorted_indices]
    y_sorted_indices = np.array(indices)[y_sorted_indices]

    # Perform Y-axis projection
    y_projection = projection_by_bboxes(boxes=y_sorted_boxes, axis=1)
    y_intervals = split_projection_profile(y_projection, 0, 1)

    if not y_intervals:
        return

    # Process each segment defined by Y-axis projection
    for y_start, y_end in zip(*y_intervals):
        # Select boxes within the current y interval
        y_interval_indices = (y_start <= y_sorted_boxes[:, 1]) & (y_sorted_boxes[:, 1] < y_end)
        y_boxes_chunk = y_sorted_boxes[y_interval_indices]
        y_indices_chunk = y_sorted_indices[y_interval_indices]

        # Sort by x_min for X-axis projection
        x_sorted_indices = y_boxes_chunk[:, 0].argsort()
        x_sorted_boxes_chunk = y_boxes_chunk[x_sorted_indices]
        x_sorted_indices_chunk = y_indices_chunk[x_sorted_indices]

        # Perform X-axis projection
        x_projection = projection_by_bboxes(boxes=x_sorted_boxes_chunk, axis=0)
        x_intervals = split_projection_profile(x_projection, 0, min_gap)

        if not x_intervals:
            continue

        # If X-axis cannot be further segmented, add current indices to results
        if len(x_intervals[0]) == 1:
            res.extend(x_sorted_indices_chunk)
            continue

        # Recursively process each segment defined by X-axis projection
        for x_start, x_end in zip(*x_intervals):
            x_interval_indices = (x_start <= x_sorted_boxes_chunk[:, 0]) & (
                x_sorted_boxes_chunk[:, 0] < x_end
            )
            recursive_yx_cut(
                x_sorted_boxes_chunk[x_interval_indices], x_sorted_indices_chunk[x_interval_indices], res
            )
            
def recursive_xy_cut(boxes: np.ndarray, indices: List[int], res: List[int], min_gap = 1):
    """
    Recursively performs X-axis projection followed by Y-axis projection to segment bounding boxes.

    Args:
        boxes: A (N, 4) array representing bounding boxes with [x_min, y_min, x_max, y_max].
        indices: A list of indices representing the position of boxes in the original data.
        res: A list to store indices of bounding boxes that meet the criteria.
    """
    # Ensure boxes and indices have the same length
    assert len(boxes) == len(indices)

    # Sort by x_min to prepare for X-axis projection
    x_sorted_indices = boxes[:, 0].argsort()
    x_sorted_boxes = boxes[x_sorted_indices]
    x_sorted_indices = np.array(indices)[x_sorted_indices]

    # Perform X-axis projection
    x_projection = projection_by_bboxes(boxes=x_sorted_boxes, axis=0)
    x_intervals = split_projection_profile(x_projection, 0, 1)

    if not x_intervals:
        return

    # Process each segment defined by X-axis projection
    for x_start, x_end in zip(*x_intervals):
        # Select boxes within the current x interval
        x_interval_indices = (x_start <= x_sorted_boxes[:, 0]) & (x_sorted_boxes[:, 0] < x_end)
        x_boxes_chunk = x_sorted_boxes[x_interval_indices]
        x_indices_chunk = x_sorted_indices[x_interval_indices]

        # Sort selected boxes by y_min to prepare for Y-axis projection
        y_sorted_indices = x_boxes_chunk[:, 1].argsort()
        y_sorted_boxes_chunk = x_boxes_chunk[y_sorted_indices]
        y_sorted_indices_chunk = x_indices_chunk[y_sorted_indices]

        # Perform Y-axis projection
        y_projection = projection_by_bboxes(boxes=y_sorted_boxes_chunk, axis=1)
        y_intervals = split_projection_profile(y_projection, 0, min_gap)

        if not y_intervals:
            continue

        # If Y-axis cannot be further segmented, add current indices to results
        if len(y_intervals[0]) == 1:
            res.extend(y_sorted_indices_chunk)
            continue

        # Recursively process each segment defined by Y-axis projection
        for y_start, y_end in zip(*y_intervals):
            y_interval_indices = (y_start <= y_sorted_boxes_chunk[:, 1]) & (y_sorted_boxes_chunk[:, 1] < y_end)
            recursive_xy_cut(
                y_sorted_boxes_chunk[y_interval_indices], y_sorted_indices_chunk[y_interval_indices], res
            )


def sort_by_xycut(block_bboxes,direction=0, min_gap=1):
    block_bboxes = np.asarray(block_bboxes).astype(int)
    res = []
    if direction == 1:
        recursive_yx_cut(block_bboxes, np.arange(len(block_bboxes)), res, min_gap)
    else:
        recursive_xy_cut(block_bboxes, np.arange(len(block_bboxes)), res, min_gap)
    return res


def get_minbox_if_overlap_by_ratio(bbox1, bbox2, ratio, smaller=True):
    """
    Determine if the overlap area between two bounding boxes exceeds a given ratio
    and return the smaller (or larger) bounding box based on the `smaller` flag.

    Args:
        bbox1, bbox2: Coordinates of bounding boxes [x_min, y_min, x_max, y_max].
        ratio (float): The overlap ratio threshold.
        smaller (bool): If True, return the smaller bounding box; otherwise, return the larger one.

    Returns:
        list or tuple: The selected bounding box or None if the overlap ratio is not exceeded.
    """
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    overlap_ratio = calculate_iou(bbox1, bbox2)

    if overlap_ratio > ratio:
        if (area1 <= area2 and smaller) or (area1 > area2 and not smaller):
            return bbox1
        else:
            return bbox2
    return None

def remove_overlaps_blocks(blocks, threshold=0.65, smaller=True):
    """
    Remove overlapping blocks based on a specified overlap ratio threshold.

    Args:
        blocks (list): List of block dictionaries, each containing a 'layout_bbox' key.
        threshold (float): Ratio threshold to determine significant overlap.
        smaller (bool): If True, the smaller block in overlap is removed.

    Returns:
        tuple: A tuple containing the updated list of blocks and a list of dropped blocks.
    """
    dropped_blocks = []
    for i, block1 in enumerate(blocks):
        for j, block2 in enumerate(blocks):
            if i >= j:
                continue
            if block1 in dropped_blocks or block2 in dropped_blocks:
                continue
            overlap_box = get_minbox_if_overlap_by_ratio(
                block1['layout_bbox'], block2['layout_bbox'], threshold, smaller=smaller
            )
            if overlap_box:
                block_to_remove = next(
                    (block for block in (block1, block2) if block['layout_bbox'] == overlap_box), None
                )
                if block_to_remove and block_to_remove not in dropped_blocks:
                    dropped_blocks.append(block_to_remove)

    for block in dropped_blocks:
        blocks.remove(block)
        block['tag'] = "block_overlap"
    return blocks, dropped_blocks

def match_bboxes(input_bboxes, input_indices, gt_bboxes, gt_indices, iou_threshold=0.5):
    """
    Match input bounding boxes to ground truth bounding boxes based on IoU.

    Args:
        input_bboxes: List of input bounding boxes.
        input_indices: List of input indices.
        gt_bboxes: List of ground truth bounding boxes.
        gt_indices: List of ground truth indices.
        iou_threshold (float): IoU threshold for matching.

    Returns:
        tuple: Matched input indices and ground truth indices.
    """
    matched_indices = []
    for gt_bbox, gt_index in zip(gt_bboxes, gt_indices):
        matched_input_indices = []
        for input_bbox, input_index in zip(input_bboxes, input_indices):
            iou = calculate_iou(input_bbox, gt_bbox)
            if iou > iou_threshold:
                matched_input_indices.append(input_index)
        if matched_input_indices:
            median_index = np.median(matched_input_indices)
        else:
            median_index = None
        matched_indices.append((median_index, gt_index))

    valid_indices = [index for index in matched_indices if index[0] is not None]
    valid_indices.sort(key=lambda x: (x[0], x[1]))
    index_mapping = {old_index: new_index+1 for new_index, old_index in enumerate(valid_indices)}

    matched_input_indices = []
    matched_gt_indices = []
    for index in matched_indices:
        result = index_mapping.get(index, None)
        if result:
            matched_input_indices.append(result)
            matched_gt_indices.append(index[1])

    return matched_input_indices, matched_gt_indices

def calculate_metrics_with_block(input_bboxes, input_indices, gt_bboxes, gt_indices):
    """
    Calculate evaluation metrics (BLEU, ARD, TAU) for matched bounding boxes.

    Args:
        input_bboxes: List of input bounding boxes.
        input_indices: List of input indices.
        gt_bboxes: List of ground truth bounding boxes.
        gt_indices: List of ground truth indices.

    Returns:
        tuple: BLEU score, ARD, and TAU values.
    """
    sorted_matched_indices, sorted_gt_indices = match_bboxes(input_bboxes, input_indices, gt_bboxes, gt_indices, iou_threshold=0.5)

    sorted_gt_indices = [index+1 for index,_ in enumerate(sorted_gt_indices)] # if ignore missing results 

    if len(sorted_gt_indices) == 0:
        return 1,0,1

    if len(sorted_gt_indices) < 4 and sorted_gt_indices == sorted_matched_indices:
        bleu_score = 1
    else:
        bleu_score = sentence_bleu([sorted_gt_indices], sorted_matched_indices)
    
    if len(sorted_gt_indices) == 0:
        ard = 0
    else:
        ard = np.mean([abs(pred - true) / true for pred, true in zip(sorted_matched_indices, sorted_gt_indices)])

    if sorted_matched_indices == sorted_gt_indices:
        tau = 1
    else:
        tau, _ = kendalltau(sorted_matched_indices, sorted_gt_indices)
        import math
        if math.isnan(tau):
            tau = 0
        
    if bleu_score < 0.95:
        print(sorted_matched_indices,sorted_gt_indices)

    return bleu_score, ard, tau

def calculate_metrics_with_page(input_data, gt_data, iou_threshold=0.5, is_order_match=True):
    """
    Calculate evaluation metrics for pages, comparing input data to ground truth data.

    Args:
        input_data: List of input page data.
        gt_data: List of ground truth page data.
        iou_threshold (float): IoU threshold for matching.
        is_order_match (bool): If True, assumes ordered page matching.

    Returns:
        tuple: Averages of BLEU score, ARD, and TAU across pages.
    """
    assert len(input_data) == len(gt_data)
    total_bleu_score = 0
    total_ard = 0
    total_tau = 0
    total_match_block_num = 0

    if not is_order_match:
        for block in input_data:
            input_bbox = block['block_bbox']
            for j, gt_block in enumerate(gt_data):
                gt_bbox = gt_block['block_bbox']
                if calculate_iou(input_bbox, gt_bbox) > iou_threshold:
                    input_bboxes = block['sub_bboxes']
                    input_indices = block['sub_indices']
                    gt_bboxes = gt_block['sub_bboxes']
                    gt_indices = gt_block['sub_indices']
                    if 0 in input_indices:
                        input_indices = [index+1 for index in input_indices]
                    if 0 in gt_indices:
                        gt_indices = [index+1 for index in gt_indices]
                    bleu_score, ard, tau = calculate_metrics_with_block(input_bboxes, input_indices, gt_bboxes, gt_indices)
                    total_bleu_score += bleu_score
                    total_ard += ard
                    total_tau += tau
                    total_match_block_num += 1
                    break
    else:
        for block_index in range(len(input_data)):
            input_bboxes = input_data[block_index]['sub_bboxes']
            gt_bboxes = gt_data[block_index]['sub_bboxes']
            input_indices = input_data[block_index]['sub_indices']
            gt_indices = gt_data[block_index]['sub_indices']
            if 0 in input_indices:
                input_indices = [index+1 for index in input_indices]
            if 0 in gt_indices:
                gt_indices = [index+1 for index in gt_indices]
            bleu_score, ard, tau = calculate_metrics_with_block(input_bboxes, input_indices, gt_bboxes, gt_indices)
            if bleu_score < 0.95:
                print(block_index,bleu_score,ard,tau)
            total_bleu_score += bleu_score
            total_ard += ard
            total_tau += tau
            total_match_block_num += 1

    return total_bleu_score / total_match_block_num, total_ard / total_match_block_num, total_tau / total_match_block_num

def paddlex_generate_input_data(data, gt_data):
    """
    Generate input data for evaluation based on layout parsing results.

    Args:
        data: Dictionary containing parsing results.

    Returns:
        list: Formatted list of input data.
    """
    parsing_result = data['layout_parsing_result']
    input_data = [{
        'block_bbox': block['block_bbox'],
        'sub_indices': [],
        'sub_bboxes': [] ,
        'page_scale': [gt_block['block_size'][0] / block['block_size'][0], gt_block['block_size'][1] / block['block_size'][1]]
    } for block,gt_block in zip(parsing_result, gt_data)]

    for block_index, block in enumerate(parsing_result):
        sub_blocks = block['sub_blocks']
        for sub_block in sub_blocks:
            if sub_block.get('index'):
                input_data[block_index]["sub_bboxes"].append(list(map(int, np.array(sub_block["layout_bbox"]) * np.array(input_data[block_index]['page_scale'] * 2))))
                input_data[block_index]["sub_indices"].append(int(sub_block["index"]))

    return input_data

def mineru_generate_input_data(data, gt_data):
    """
    Generate input data for evaluation based on layout parsing results.

    Args:
        data: Dictionary containing parsing results.
        gt_data: Ground truth data for comparison.

    Returns:
        list: Formatted list of input data.
    """
    parsing_result = data['pdf_info']
    parsing_result = [{
        "block_bbox": [0, 0, 2550, 2550],  # Page boundary bounding box
        "sub_blocks": block['para_blocks'],
        "block_size": block['page_size']
    } for block in parsing_result]

    input_data = [{
        'block_bbox': block['block_bbox'],
        'sub_indices': [],
        'sub_bboxes': [],
        'page_scale': [gt_block['block_size'][0] / block['block_size'][0], gt_block['block_size'][1] / block['block_size'][1]]
    } for block, gt_block in zip(parsing_result, gt_data)]

    for block_index, block in enumerate(parsing_result):
        sub_blocks = block['sub_blocks']
        sub_blocks = sorted(sub_blocks, key=lambda x: (x['index'], x['bbox'][1], x['bbox'][0]))
        for i, sub_block in enumerate(sub_blocks):
            input_data[block_index]["sub_bboxes"].append(list(map(int, np.array(sub_block["bbox"]) * np.array(input_data[block_index]['page_scale'] * 2))))
            input_data[block_index]["sub_indices"].append(i + 1)

    return input_data

def load_data_from_json(path):
    """
    Load data from a JSON file.

    Args:
        path (str): File path to the JSON file.

    Returns:
        dict: Parsed data from the JSON file.
    """
    with open(path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


