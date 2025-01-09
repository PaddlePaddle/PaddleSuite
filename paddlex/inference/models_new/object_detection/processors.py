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

from typing import List, Sequence, Tuple, Union, Optional

import cv2
import numpy as np
from numpy import ndarray

from ..common import Resize as CommonResize
from ..common import Normalize as CommonNormalize
from ...common.reader import ReadImage as CommonReadImage

Boxes = List[dict]
Number = Union[int, float]


class ReadImage(CommonReadImage):
    """Reads images from a list of raw image data or file paths."""

    def __call__(self, raw_imgs: List[Union[ndarray, str]]) -> List[dict]:
        """Processes the input list of raw image data or file paths and returns a list of dictionaries containing image information.

        Args:
            raw_imgs (List[Union[ndarray, str]]): A list of raw image data (numpy ndarrays) or file paths (strings).

        Returns:
            List[dict]: A list of dictionaries, each containing image information.
        """
        out_datas = []
        for raw_img in raw_imgs:
            data = dict()
            if isinstance(raw_img, str):
                data["img_path"] = raw_img
            img = self.read(raw_img)
            data["img"] = img
            data["ori_img"] = img
            data["img_size"] = [img.shape[1], img.shape[0]]  # [size_w, size_h]
            data["ori_img_size"] = [img.shape[1], img.shape[0]]  # [size_w, size_h]

            out_datas.append(data)

        return out_datas


class Resize(CommonResize):
    def __call__(self, datas: List[dict]) -> List[dict]:
        """
        Args:
            datas (List[dict]): A list of dictionaries, each containing image data with key 'img'.

        Returns:
            List[dict]: A list of dictionaries with updated image data, including resized images,
                original image sizes, resized image sizes, and scale factors.
        """
        for data in datas:
            ori_img = data["img"]
            if "ori_img_size" not in data:
                data["ori_img_size"] = [ori_img.shape[1], ori_img.shape[0]]
            ori_img_size = data["ori_img_size"]

            img = self.resize(ori_img)
            data["img"] = img

            img_size = [img.shape[1], img.shape[0]]
            data["img_size"] = img_size  # [size_w, size_h]

            data["scale_factors"] = [  # [w_scale, h_scale]
                img_size[0] / ori_img_size[0],
                img_size[1] / ori_img_size[1],
            ]

        return datas


class Normalize(CommonNormalize):
    """Normalizes images in a list of dictionaries containing image data"""

    def apply(self, img: ndarray) -> ndarray:
        """Applies normalization to a single image."""
        old_type = img.dtype
        # XXX: If `old_type` has higher precision than float32,
        # we will lose some precision.
        img = img.astype("float32", copy=False)
        img *= self.scale
        img -= self.mean
        img /= self.std
        if self.preserve_dtype:
            img = img.astype(old_type, copy=False)
        return img

    def __call__(self, datas: List[dict]) -> List[dict]:
        """Normalizes images in a list of dictionaries. Iterates over each dictionary,
        applies normalization to the 'img' key, and returns the modified list.
        """
        for data in datas:
            data["img"] = self.apply(data["img"])
        return datas


class ToCHWImage:
    """Converts images in a list of dictionaries from HWC to CHW format."""

    def __call__(self, datas: List[dict]) -> List[dict]:
        """Converts the image data in the list of dictionaries from HWC to CHW format in-place.

        Args:
            datas (List[dict]): A list of dictionaries, each containing an image tensor in 'img' key with HWC format.

        Returns:
            List[dict]: The same list of dictionaries with the image tensors converted to CHW format.
        """
        for data in datas:
            data["img"] = data["img"].transpose((2, 0, 1))
        return datas


class ToBatch:
    """
    Class for batch processing of data dictionaries.

    Args:
        ordered_required_keys (Optional[Tuple[str]]): A tuple of keys that need to be present in the input data dictionaries in a specific order.
    """

    def __init__(self, ordered_required_keys: Optional[Tuple[str]] = None):
        self.ordered_required_keys = ordered_required_keys

    def apply(
        self, datas: List[dict], key: str, dtype: np.dtype = np.float32
    ) -> np.ndarray:
        """
        Apply batch processing to a list of data dictionaries.

        Args:
            datas (List[dict]): A list of data dictionaries to process.
            key (str): The key in the data dictionaries to extract and batch.
            dtype (np.dtype): The desired data type of the output array (default is np.float32).

        Returns:
            np.ndarray: A numpy array containing the batched data.

        Raises:
            KeyError: If the specified key is not found in any of the data dictionaries.
        """
        if key == "img_size":
            # [h, w] size for det models
            img_sizes = [data[key][::-1] for data in datas]
            return np.stack(img_sizes, axis=0).astype(dtype=dtype, copy=False)

        elif key == "scale_factors":
            # [h, w] scale factors for det models, default [1.0, 1.0]
            scale_factors = [data.get(key, [1.0, 1.0])[::-1] for data in datas]
            return np.stack(scale_factors, axis=0).astype(dtype=dtype, copy=False)

        else:
            return np.stack([data[key] for data in datas], axis=0).astype(
                dtype=dtype, copy=False
            )

    def __call__(self, datas: List[dict]) -> Sequence[ndarray]:
        return [self.apply(datas, key) for key in self.ordered_required_keys]


class DetPad:
    """
    Pad image to a specified size.
    Args:
        size (list[int]): image target size
        fill_value (list[float]): rgb value of pad area, default (114.0, 114.0, 114.0)
    """

    def __init__(
        self,
        size: List[int],
        fill_value: List[Union[int, float]] = [114.0, 114.0, 114.0],
    ):
        super().__init__()
        if isinstance(size, int):
            size = [size, size]
        self.size = size
        self.fill_value = fill_value

    def apply(self, img: ndarray) -> ndarray:
        im = img
        im_h, im_w = im.shape[:2]
        h, w = self.size
        if h == im_h and w == im_w:
            return im

        canvas = np.ones((h, w, 3), dtype=np.float32)
        canvas *= np.array(self.fill_value, dtype=np.float32)
        canvas[0:im_h, 0:im_w, :] = im.astype(np.float32)
        return canvas

    def __call__(self, datas: List[dict]) -> List[dict]:
        for data in datas:
            data["img"] = self.apply(data["img"])
        return datas


class PadStride:
    """padding image for model with FPN , instead PadBatch(pad_to_stride, pad_gt) in original config
    Args:
        stride (bool): model with FPN need image shape % stride == 0
    """

    def __init__(self, stride: int = 0):
        super().__init__()
        self.coarsest_stride = stride

    def apply(self, img: ndarray):
        """
        Args:
            im (np.ndarray): image (np.ndarray)
        Returns:
            im (np.ndarray):  processed image (np.ndarray)
        """
        im = img
        coarsest_stride = self.coarsest_stride
        if coarsest_stride <= 0:
            return img
        im_c, im_h, im_w = im.shape
        pad_h = int(np.ceil(float(im_h) / coarsest_stride) * coarsest_stride)
        pad_w = int(np.ceil(float(im_w) / coarsest_stride) * coarsest_stride)
        padding_im = np.zeros((im_c, pad_h, pad_w), dtype=np.float32)
        padding_im[:, :im_h, :im_w] = im
        return padding_im

    def __call__(self, datas: List[dict]) -> List[dict]:
        for data in datas:
            data["img"] = self.apply(data["img"])
        return datas


def rotate_point(pt: List[float], angle_rad: float) -> List[float]:
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


def _get_3rd_point(a: ndarray, b: ndarray) -> ndarray:
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
    center: ndarray,
    input_size: Union[Number, Tuple[Number, Number], ndarray],
    rot: float,
    output_size: ndarray,
    shift: Tuple[float, float] = (0.0, 0.0),
    inv: bool = False,
):
    """Get the affine transform matrix, given the center/scale/rot/output_size.
    Args:
        center (np.ndarray[2, ]): Center of the bounding box (x, y).
        input_size (np.ndarray[2, ]): Scale of the bounding box
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
    if not isinstance(input_size, (ndarray, list)):
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


class WarpAffine:
    """Apply warp affine transformation to the image based on the given parameters.

    Args:
        keep_res (bool): Whether to keep the original resolution aspect ratio during transformation.
        pad (int): Padding value used when keep_res is True.
        input_h (int): Target height for the input image when keep_res is False.
        input_w (int): Target width for the input image when keep_res is False.
        scale (float): Scale factor for resizing.
        shift (float): Shift factor for transformation.
        down_ratio (int): Downsampling ratio for the output image.
    """

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

    def apply(self, img: ndarray):

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

        return inp

    def __call__(self, datas: List[dict]) -> List[dict]:

        for data in datas:
            ori_img = data["img"]
            if "ori_img_size" not in data:
                data["ori_img_size"] = [ori_img.shape[1], ori_img.shape[0]]
            ori_img_size = data["ori_img_size"]

            img = self.apply(ori_img)
            data["img"] = img

            img_size = [img.shape[1], img.shape[0]]
            data["img_size"] = img_size  # [size_w, size_h]

            data["scale_factors"] = [  # [w_scale, h_scale]
                img_size[0] / ori_img_size[0],
                img_size[1] / ori_img_size[1],
            ]

        return datas


def compute_iou(box1: List[Number], box2: List[Number]) -> float:
    """Compute the Intersection over Union (IoU) of two bounding boxes.

    Args:
        box1 (List[Number]): Coordinates of the first bounding box in format [x1, y1, x2, y2].
        box2 (List[Number]): Coordinates of the second bounding box in format [x1, y1, x2, y2].

    Returns:
        float: The IoU of the two bounding boxes.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou


def is_box_mostly_inside(
    inner_box: List[Number], outer_box: List[Number], threshold: float = 0.9
) -> bool:
    """Determine if one bounding box is mostly inside another bounding box.

    Args:
        inner_box (List[Number]): Coordinates of the inner bounding box in format [x1, y1, x2, y2].
        outer_box (List[Number]): Coordinates of the outer bounding box in format [x1, y1, x2, y2].
        threshold (float): The threshold for determining if the inner box is mostly inside the outer box (default is 0.9).

    Returns:
        bool: True if the ratio of the intersection area to the inner box area is greater than or equal to the threshold, False otherwise.
    """
    x1 = max(inner_box[0], outer_box[0])
    y1 = max(inner_box[1], outer_box[1])
    x2 = min(inner_box[2], outer_box[2])
    y2 = min(inner_box[3], outer_box[3])
    inter_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    inner_box_area = (inner_box[2] - inner_box[0] + 1) * (
        inner_box[3] - inner_box[1] + 1
    )
    return (inter_area / inner_box_area) >= threshold


def restructured_boxes(
    boxes: ndarray, labels: List[str], img_size: Tuple[int, int]
) -> Boxes:
    """
    Restructure the given bounding boxes and labels based on the image size.

    Args:
        boxes (ndarray): A 2D array of bounding boxes with each box represented as [cls_id, score, xmin, ymin, xmax, ymax].
        labels (List[str]): A list of class labels corresponding to the class ids.
        img_size (Tuple[int, int]): A tuple representing the width and height of the image.

    Returns:
        Boxes: A list of dictionaries, each containing 'cls_id', 'label', 'score', and 'coordinate' keys.
    """
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


def restructured_rotated_boxes(
    boxes: ndarray, labels: List[str], img_size: Tuple[int, int]
) -> Boxes:
    """
    Restructure the given rotated bounding boxes and labels based on the image size.

    Args:
        boxes (ndarray): A 2D array of rotated bounding boxes with each box represented as [cls_id, score, x1, y1, x2, y2, x3, y3, x4, y4].
        labels (List[str]): A list of class labels corresponding to the class ids.
        img_size (Tuple[int, int]): A tuple representing the width and height of the image.

    Returns:
        Boxes: A list of dictionaries, each containing 'cls_id', 'label', 'score', and 'coordinate' keys.
    """
    box_list = []
    w, h = img_size

    assert boxes.shape[1] == 10, "The shape of rotated boxes should be [N, 10]"
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


def non_max_suppression(
    boxes: ndarray, scores: ndarray, iou_threshold: float
) -> List[int]:
    """
    Perform non-maximum suppression to remove redundant overlapping boxes with
    lower scores. This function is commonly used in object detection tasks.

    Parameters:
    boxes (ndarray): An array of shape (N, 4) representing the bounding boxes.
        Each row is in the format [x1, y1, x2, y2].
    scores (ndarray): An array of shape (N,) containing the scores for each box.
    iou_threshold (float): The Intersection over Union (IoU) threshold to use
        for suppressing overlapping boxes.

    Returns:
    List[int]: A list of indices representing the indices of the boxes to keep.
    """
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


class DetPostProcess:
    """Save Result Transform

    This class is responsible for post-processing detection results, including
    thresholding, non-maximum suppression (NMS), and restructuring the boxes
    based on the input type (normal or rotated object detection).
    """

    def __init__(
        self,
        threshold: float = 0.5,
        labels: Optional[List[str]] = None,
        layout_postprocess: bool = False,
    ) -> None:
        """Initialize the DetPostProcess class.

        Args:
            threshold (float, optional): The threshold to apply to the detection scores. Defaults to 0.5.
            labels (Optional[List[str]], optional): The list of labels for the detection categories. Defaults to None.
            layout_postprocess (bool, optional): Whether to apply layout post-processing. Defaults to False.
        """
        super().__init__()
        self.threshold = threshold
        self.labels = labels
        self.layout_postprocess = layout_postprocess

    def apply(self, boxes: ndarray, img_size, threshold: Union[float, dict]) -> Boxes:
        """Apply post-processing to the detection boxes.

        Args:
            boxes (ndarray): The input detection boxes with scores.
            img_size (tuple): The original image size.

        Returns:
            Boxes: The post-processed detection boxes.
        """
        if isinstance(threshold, float):
            expect_boxes = (boxes[:, 1] > threshold) & (boxes[:, 0] > -1)
            boxes = boxes[expect_boxes, :]
        elif isinstance(threshold, dict):
            category_filtered_boxes = []
            for cat_id in np.unique(boxes[:, 0]):
                category_boxes = boxes[boxes[:, 0] == cat_id]
                category_scores = category_boxes[:, 1]
                category_threshold = threshold.get(int(cat_id), 0.5)
                selected_indices = category_scores > category_threshold
                category_filtered_boxes.append(category_boxes[selected_indices])
            boxes = (
                np.vstack(category_filtered_boxes)
                if category_filtered_boxes
                else np.array([])
            )

        if self.layout_postprocess:
            filtered_boxes = []
            ### Layout postprocess for NMS
            for cat_id in np.unique(boxes[:, 0]):
                category_boxes = boxes[boxes[:, 0] == cat_id]
                category_scores = category_boxes[:, 1]
                if len(category_boxes) > 0:
                    nms_indices = non_max_suppression(
                        category_boxes[:, 2:], category_scores, 0.5
                    )
                    category_boxes = category_boxes[nms_indices]
                    keep_boxes = []
                    for i, box in enumerate(category_boxes):
                        if all(
                            not is_box_mostly_inside(box[2:], other_box[2:])
                            for j, other_box in enumerate(category_boxes)
                            if i != j
                        ):
                            keep_boxes.append(box)
                    filtered_boxes.extend(keep_boxes)
            boxes = np.array(filtered_boxes)
            ### Layout postprocess for removing boxes inside image category box
            if self.labels and "image" in self.labels:
                image_cls_id = self.labels.index("image")
                if len(boxes) > 0:
                    image_boxes = boxes[boxes[:, 0] == image_cls_id]
                    other_boxes = boxes[boxes[:, 0] != image_cls_id]
                    to_keep = []
                    for box in other_boxes:
                        keep = True
                        for img_box in image_boxes:
                            if (
                                box[2] >= img_box[2]
                                and box[3] >= img_box[3]
                                and box[4] <= img_box[4]
                                and box[5] <= img_box[5]
                            ):
                                keep = False
                                break
                        if keep:
                            to_keep.append(box)
                    boxes = (
                        np.vstack([image_boxes, to_keep]) if to_keep else image_boxes
                    )
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

        if boxes.shape[1] == 6:
            """For Normal Object Detection"""
            boxes = restructured_boxes(boxes, self.labels, img_size)
        elif boxes.shape[1] == 10:
            """Adapt For Rotated Object Detection"""
            boxes = restructured_rotated_boxes(boxes, self.labels, img_size)
        else:
            """Unexpected Input Box Shape"""
            raise ValueError(
                f"The shape of boxes should be 6 or 10, instead of {boxes.shape[1]}"
            )
        return boxes

    def __call__(
        self,
        batch_outputs: List[dict],
        datas: List[dict],
        threshold: Optional[Union[float, dict]] = None,
    ) -> List[Boxes]:
        """Apply the post-processing to a batch of outputs.

        Args:
            batch_outputs (List[dict]): The list of detection outputs.
            datas (List[dict]): The list of input data.

        Returns:
            List[Boxes]: The list of post-processed detection boxes.
        """
        outputs = []
        for data, output in zip(datas, batch_outputs):
            boxes = self.apply(
                output["boxes"],
                data["ori_img_size"],
                threshold if threshold is not None else self.threshold,
            )
            outputs.append(boxes)
        return outputs
