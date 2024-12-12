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

import os
import cv2

import numpy as np
from ...utils.io import ImageReader
from ..base import BaseComponent


def restructured_boxes(boxes, labels, img_size):

    box_list = []
    w, h = img_size

    for box in boxes:
        xmin, ymin, xmax, ymax = box[2:]
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)
        box_list.append(
            {
                "cls_id": int(box[0]),
                "label": labels[int(box[0])],
                "score": float(box[1]),
                "coordinate": [xmin, ymin, xmax, ymax],
            }
        )

    return box_list


def restructured_rotated_boxes(boxes, labels, img_size):

    box_list = []
    w, h = img_size

    assert boxes.shape[1] == 10, 'The shape of rotated boxes should be [N, 10]'
    for box in boxes:
        x1, y1, x2, y2, x3, y3, x4, y4 = box[2:]
        x1 = min(max(0, x1), w)
        y1 = min(max(0, y1), h)
        x2 = min(max(0, x2), w)
        y2 = min(max(0, y2), h)
        x3 = min(max(0, x3), w)
        y3 = min(max(0, y3), h)
        x4 = min(max(0, x4), w)
        y4 = min(max(0, y4), h)
        box_list.append(
            {
                "cls_id": int(box[0]),
                "label": labels[int(box[0])],
                "score": float(box[1]),
                "coordinate": [x1, y1, x2, y2, x3, y3, x4, y4],
            }
        )

    return box_list


def rotate_point(pt, angle_rad):
    """Rotate a point by an angle.
    Args:
        pt (list[float]): 2 dimensional point to be rotated
        angle_rad (float): rotation angle by radian
    Returns:
        list[float]: Rotated point.
    """
    assert len(pt) == 2
    sn, cs = np.sin(angle_rad), np.cos(angle_rad)
    new_x = pt[0] * cs - pt[1] * sn
    new_y = pt[0] * sn + pt[1] * cs
    rotated_pt = [new_x, new_y]

    return rotated_pt


def _get_3rd_point(a, b):
    """To calculate the affine matrix, three pairs of points are required. This
    function is used to get the 3rd point, given 2D points a & b.
    The 3rd point is defined by rotating vector `a - b` by 90 degrees
    anticlockwise, using b as the rotation center.
    Args:
        a (np.ndarray): point(x,y)
        b (np.ndarray): point(x,y)
    Returns:
        np.ndarray: The 3rd point.
    """
    assert len(a) == 2
    assert len(b) == 2
    direction = a - b
    third_pt = b + np.array([-direction[1], direction[0]], dtype=np.float32)

    return third_pt


def get_affine_transform(
    center, input_size, rot, output_size, shift=(0.0, 0.0), inv=False
):
    """Get the affine transform matrix, given the center/scale/rot/output_size.
    Args:
        center (np.ndarray[2, ]): Center of the bounding box (x, y).
        scale (np.ndarray[2, ]): Scale of the bounding box
            wrt [width, height].
        rot (float): Rotation angle (degree).
        output_size (np.ndarray[2, ]): Size of the destination heatmaps.
        shift (0-100%): Shift translation ratio wrt the width/height.
            Default (0., 0.).
        inv (bool): Option to inverse the affine transform direction.
            (inv=False: src->dst or inv=True: dst->src)
    Returns:
        np.ndarray: The transform matrix.
    """
    assert len(center) == 2
    assert len(output_size) == 2
    assert len(shift) == 2
    if not isinstance(input_size, (np.ndarray, list)):
        input_size = np.array([input_size, input_size], dtype=np.float32)
    scale_tmp = input_size

    shift = np.array(shift)
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = rotate_point([0.0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0.0, dst_w * -0.5])

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    src[2, :] = _get_3rd_point(src[0, :], src[1, :])

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


class WarpAffine(BaseComponent):
    """Warp affine the image"""

    INPUT_KEYS = ["img"]
    OUTPUT_KEYS = ["img", "img_size", "scale_factors"]
    DEAULT_INPUTS = {"img": "img"}
    DEAULT_OUTPUTS = {
        "img": "img",
        "img_size": "img_size",
        "scale_factors": "scale_factors",
    }

    def __init__(
        self,
        keep_res=False,
        pad=31,
        input_h=512,
        input_w=512,
        scale=0.4,
        shift=0.1,
        down_ratio=4,
    ):
        super().__init__()
        self.keep_res = keep_res
        self.pad = pad
        self.input_h = input_h
        self.input_w = input_w
        self.scale = scale
        self.shift = shift
        self.down_ratio = down_ratio

    def apply(self, img):

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        h, w = img.shape[:2]

        if self.keep_res:
            # True in detection eval/infer
            input_h = (h | self.pad) + 1
            input_w = (w | self.pad) + 1
            s = np.array([input_w, input_h], dtype=np.float32)
            c = np.array([w // 2, h // 2], dtype=np.float32)

        else:
            # False in centertrack eval_mot/eval_mot
            s = max(h, w) * 1.0
            input_h, input_w = self.input_h, self.input_w
            c = np.array([w / 2.0, h / 2.0], dtype=np.float32)

        trans_input = get_affine_transform(c, s, 0, [input_w, input_h])
        img = cv2.resize(img, (w, h))
        inp = cv2.warpAffine(
            img, trans_input, (input_w, input_h), flags=cv2.INTER_LINEAR
        )

        if not self.keep_res:
            out_h = input_h // self.down_ratio
            out_w = input_w // self.down_ratio
            trans_output = get_affine_transform(c, s, 0, [out_w, out_h])

        im_scale_w, im_scale_h = [input_w / w, input_h / h]

        return {
            "img": inp,
            "img_size": [inp.shape[1], inp.shape[0]],
            "scale_factors": [im_scale_w, im_scale_h],
        }


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou


def is_box_mostly_inside(inner_box, outer_box, threshold=0.9):
    x1 = max(inner_box[0], outer_box[0])
    y1 = max(inner_box[1], outer_box[1])
    x2 = min(inner_box[2], outer_box[2])
    y2 = min(inner_box[3], outer_box[3])
    inter_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    inner_box_area = (inner_box[2] - inner_box[0] + 1) * (inner_box[3] - inner_box[1] + 1)
    return (inter_area / inner_box_area) >= threshold


def non_max_suppression(boxes, scores, iou_threshold):
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return keep


class DetPostProcess(BaseComponent):
    """Save Result Transform"""

    INPUT_KEYS = ["input_path", "boxes", "img_size"]
    OUTPUT_KEYS = ["boxes"]
    DEAULT_INPUTS = {"boxes": "boxes", "img_size": "ori_img_size"}
    DEAULT_OUTPUTS = {"boxes": "boxes"}

    def __init__(self, threshold=0.5, labels=None, layout_postprocess=False):
        super().__init__()
        self.threshold = threshold
        self.labels = labels
        self.layout_postprocess = layout_postprocess

    def apply(self, boxes, img_size):
        """apply"""
        filtered_boxes = []
        if isinstance(self.threshold, float):
            expect_boxes = (boxes[:, 1] > self.threshold) & (boxes[:, 0] > -1)
            boxes = boxes[expect_boxes, :]
        elif isinstance(self.threshold, dict):
            category_filtered_boxes = []
            for cat_id in np.unique(boxes[:, 0]):
                category_boxes = boxes[boxes[:, 0] == cat_id]
                category_scores = category_boxes[:, 1]
                category_threshold = self.threshold.get(int(cat_id), 0.5)
                selected_indices = category_scores > category_threshold
                category_filtered_boxes.append(category_boxes[selected_indices])
            boxes = np.vstack(category_filtered_boxes) if category_filtered_boxes else np.array([])

        if self.layout_postprocess:
            ### Layout postprocess for NMS
            for cat_id in np.unique(boxes[:, 0]):
                category_boxes = boxes[boxes[:, 0] == cat_id]
                category_scores = category_boxes[:, 1]
                if len(category_boxes) > 0:
                    nms_indices = non_max_suppression(category_boxes[:, 2:], category_scores, 0.5)
                    category_boxes = category_boxes[nms_indices]
                    keep_boxes = []
                    for i, box in enumerate(category_boxes):
                        if all(not is_box_mostly_inside(box[2:], other_box[2:]) for j, other_box in enumerate(category_boxes) if i != j):
                            keep_boxes.append(box)
                    filtered_boxes.extend(keep_boxes)
            boxes = np.array(filtered_boxes)
            ### Layout postprocess for removing boxes inside image category box
            if self.labels and "image" in self.labels:
                image_cls_id = self.labels.index('image')
                if len(boxes) > 0:
                    image_boxes = boxes[boxes[:, 0] == image_cls_id]
                    other_boxes = boxes[boxes[:, 0] != image_cls_id]
                    to_keep = []
                    for box in other_boxes:
                        keep = True
                        for img_box in image_boxes:
                            if (box[2] >= img_box[2] and box[3] >= img_box[3] and
                                box[4] <= img_box[4] and box[5] <= img_box[5]):
                                keep = False
                                break
                        if keep:
                            to_keep.append(box)
                    boxes = np.vstack([image_boxes, to_keep]) if to_keep else image_boxes
            ### Layout postprocess for overlaps
            final_boxes = []
            while len(boxes) > 0:
                current_box = boxes[0]
                current_score = current_box[1]
                overlaps = [current_box]
                non_overlaps = []
                for other_box in boxes[1:]:
                    iou = compute_iou(current_box[2:], other_box[2:])
                    if iou > 0.95:
                        if other_box[1] > current_score:
                            overlaps.append(other_box)
                    else:
                        non_overlaps.append(other_box)
                best_box = max(overlaps, key=lambda x: x[1])
                final_boxes.append(best_box)
                boxes = np.array(non_overlaps)
            boxes = np.array(final_boxes)

        boxes = restructured_boxes(boxes, self.labels, img_size)
        result = {"boxes": boxes}
        return result


class CropByBoxes(BaseComponent):
    """Crop Image by Box"""

    YIELD_BATCH = False
    INPUT_KEYS = ["input_path", "boxes"]
    OUTPUT_KEYS = ["img", "box", "label"]
    DEAULT_INPUTS = {"input_path": "input_path", "boxes": "boxes"}
    DEAULT_OUTPUTS = {"img": "img", "box": "box", "label": "label"}

    def __init__(self):
        super().__init__()
        self._reader = ImageReader(backend="opencv")

    def apply(self, input_path, boxes):
        output_list = []
        img = self._reader.read(input_path)
        for bbox in boxes:
            label_id = bbox["cls_id"]
            box = bbox["coordinate"]
            label = bbox.get("label", label_id)
            xmin, ymin, xmax, ymax = [int(i) for i in box]
            img_crop = img[ymin:ymax, xmin:xmax]
            output_list.append({"img": img_crop, "box": box, "label": label})

        return output_list


class DetPad(BaseComponent):

    INPUT_KEYS = "img"
    OUTPUT_KEYS = "img"
    DEAULT_INPUTS = {"img": "img"}
    DEAULT_OUTPUTS = {"img": "img"}

    def __init__(self, size, fill_value=[114.0, 114.0, 114.0]):
        """
        Pad image to a specified size.
        Args:
            size (list[int]): image target size
            fill_value (list[float]): rgb value of pad area, default (114.0, 114.0, 114.0)
        """

        super().__init__()
        if isinstance(size, int):
            size = [size, size]
        self.size = size
        self.fill_value = fill_value

    def apply(self, img):
        im = img
        im_h, im_w = im.shape[:2]
        h, w = self.size
        if h == im_h and w == im_w:
            return {"img": im}

        canvas = np.ones((h, w, 3), dtype=np.float32)
        canvas *= np.array(self.fill_value, dtype=np.float32)
        canvas[0:im_h, 0:im_w, :] = im.astype(np.float32)
        return {"img": canvas}
