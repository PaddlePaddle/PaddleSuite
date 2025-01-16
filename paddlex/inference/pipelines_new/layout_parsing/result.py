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

import inspect
import os
from pathlib import Path
import json
import numpy as np
import uuid
import cv2
import os
import json
from PIL import Image, ImageDraw, ImageFont
from .utils import sort_by_xycut
from .utils import calculate_metrics_with_page
from ..components.utils.mixin import JsonMixin, ImgMixin, StrMixin, MarkdownMixin

class LayoutParsingResult(dict, StrMixin, JsonMixin, MarkdownMixin, ImgMixin):
    """Layout Parsing Result"""

    def __init__(self, data,page_id=None,src_input_name=None) -> None:
        """Initializes a new instance of the class with the specified data."""
        super().__init__(data)
        self._show_funcs = []
        self.page_id = page_id
        if isinstance(src_input_name, list):
            self.src_input_name = src_input_name[page_id]
        else:
            self.src_input_name = src_input_name
        StrMixin.__init__(self)
        JsonMixin.__init__(self)
        MarkdownMixin.__init__(self)
        self.is_ordered = False
        
    
    def save_all(self, save_path):
        for func in self._show_funcs:
            signature = inspect.signature(func)
            if "save_path" in signature.parameters:
                func(save_path=save_path)
            else:
                func()

    def save_results(self, save_path: str) -> None:
        """Save the layout parsing results to the specified directory.

        Args:
            save_path (str): The directory path to save the results.
        """

        if not os.path.isdir(save_path):
            return
        save_path = os.path.join(save_path, "images")
        img_id = self["img_id"]
        layout_det_res = self["layout_det_res"]
        save_img_path = Path(save_path) / f"layout_det_result_img{img_id}.jpg"
        layout_det_res.save_to_img(save_img_path)

        input_params = self["input_params"]
        if input_params["use_doc_preprocessor"]:
            save_img_path = Path(save_path) / f"doc_preprocessor_result_img{img_id}.jpg"
            self["doc_preprocessor_res"].save_to_img(save_img_path)

        if input_params["use_general_ocr"]:
            save_img_path = (
                Path(save_path) / f"text_paragraphs_ocr_result_img{img_id}.jpg"
            )
            self["text_paragraphs_ocr_res"].save_to_img(save_img_path)

        if input_params["use_general_ocr"] or input_params["use_table_recognition"]:
            save_img_path = Path(save_path) / f"overall_ocr_result_img{img_id}.jpg"
            self["overall_ocr_res"].save_to_img(save_img_path)

        if input_params["use_table_recognition"]:
            for tno in range(len(self["table_res_list"])):
                table_res = self["table_res_list"][tno]
                table_region_id = table_res["table_region_id"]
                save_img_path = (
                    Path(save_path)
                    / f"table_res_cell_img{img_id}_region{table_region_id}.jpg"
                )
                table_res.save_to_img(save_img_path)
                save_html_path = (
                    Path(save_path)
                    / f"table_res_img{img_id}_region{table_region_id}.html"
                )
                table_res.save_to_html(save_html_path)
                save_xlsx_path = (
                    Path(save_path)
                    / f"table_res_img{img_id}_region{table_region_id}.xlsx"
                )
                table_res.save_to_xlsx(save_xlsx_path)

        if input_params["use_seal_recognition"]:
            for sno in range(len(self["seal_res_list"])):
                seal_res = self["seal_res_list"][sno]
                seal_region_id = seal_res["seal_region_id"]
                save_img_path = (
                    Path(save_path) / f"seal_res_img{img_id}_region{seal_region_id}.jpg"
                )
                seal_res.save_to_img(save_img_path)
        return

    def get_target_name(self, save_path):
        if self.src_input_name.endswith(".pdf"):
            save_path = (
                Path(save_path)
                / f"{Path(self.src_input_name).stem}_pdf"
                / Path("page_{:04d}".format(self.page_id + 1))
            )
        else:
            save_path = Path(save_path) / f"{Path(self.src_input_name).stem}"
        return save_path

    def save_to_json(self, save_path):
        if not save_path.lower().endswith(("json")):
            save_path = os.path.join(save_path, "jsons")
            save_path = self.get_target_name(save_path)
            save_dir = Path(save_path).parent.parent.parent
        else:
            save_path = Path(save_path).stem
            save_dir = Path(save_path).parent.parent

        self._recursive_img_array2path(self, save_dir ,labels=['img','input_img','output_img','input_path','doc_preprocessor_image']) # markwon path

        image_res_list = []
        parsing_result = self['layout_parsing_result']
        for block in parsing_result:
            sub_blocks = block['sub_blocks']
            for sub_block in sub_blocks:
                if sub_block['label'] == 'image':
                    image_res_list.append(sub_block['image']['img'])
                elif sub_block['label'] == 'chart':
                    image_res_list.append(sub_block['chart']['img'])
    
        self['input_path'] = self._img_array2path(self['image_array'],save_dir)
        del self['image_array']
        # del self['overall_ocr_res']
        # del self['text_paragraphs_ocr_res']
        del self['doc_preprocessor_res']
        # del self['layout_det_res']
        self['image_res_list'] = image_res_list

        if not str(save_path).endswith(".json"):
            save_path = "{}.json".format(save_path)
        super().save_to_json(save_path)

    def save_to_pdf_order(self, save_path, is_eval=False, font_path="/workspace/shuailiu35/Roboto-Bold.ttf", is_only_xycut=False):
        """
        Save the layout ordering to an image file.

        Args:
            save_path (str or Path): The path where the image should be saved.
            is_eval (bool): Flag indicating whether to evaluate specific conditions.
            font_path (str): Path to the font file used for drawing text.
            is_only_xycut (bool): Flag indicating whether to apply only XY cut.

        Returns:
            None
        """
        save_path = os.path.join(save_path, "ordering")
        save_path = Path(save_path)
        if save_path.suffix.lower() not in (".jpg", ".png"):
            save_path = Path(self.get_target_name(str(save_path)))
        else:
            save_path = save_path.with_suffix('')
        ordering_image_path = save_path.parent / f"{save_path.stem}_ordering.jpg"

        # Open the input image
        try:
            # image = Image.open(self['input_path'])
            image = Image.fromarray(self['image_array'])
        except IOError as e:
            print(f"Error opening image: {e}")
            return

        draw = ImageDraw.Draw(image,'RGBA')
        font_size = 20

        # Load the specified font or use default if not found
        try:
            font = ImageFont.truetype(font_path, font_size)
        except IOError:
            print("Font not found, using default font.")
            font = ImageFont.load_default()

        parsing_result = self['layout_parsing_result']
        for block_index, block in enumerate(parsing_result):
            self._get_layout_ordering(
                block_index=block_index,
                no_mask_labels=['text', 'formula', 'algorithm', "reference", "content", "abstract"],
                is_eval=is_eval,
                is_only_xycut=is_only_xycut
            )

            # Draw bounding boxes and indices on the image
            sub_blocks = parsing_result[block_index]['sub_blocks']
            for sub_block in sub_blocks:
                bbox = sub_block['layout_bbox']
                index = sub_block.get('index',None)
                label = sub_block['sub_label']
                fill_color = self._get_show_color(label)
                # draw.rectangle(bbox, outline="blue", width=4)
                draw.rectangle(bbox, fill=fill_color)
                if index is not None:
                    text_position = (bbox[2]+2, bbox[1] - 10)
                    draw.text(text_position, str(index), fill="red", font=font)

        # Ensure the directory exists and save the image
        ordering_image_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving ordering image to {ordering_image_path}")
        image.save(str(ordering_image_path))

    def save_to_markdown(self, save_path, is_eval=True, is_only_xycut=False):
        """
        Save the parsing result to a Markdown file.

        Args:
            save_path (str or Path): The path where the Markdown file should be saved.
            is_eval (bool): Flag indicating whether to evaluate specific conditions.
            is_only_xycut (bool): Flag indicating whether to apply only XY cut.

        Returns:
            None
        """
        save_path = Path(save_path)
        if not save_path.suffix.lower() == ".md":
            save_path = save_path / f"layout_parsing_result.md"

        parsing_result = self['layout_parsing_result']
        for block_index, _ in enumerate(parsing_result):
            self._get_layout_ordering(
                block_index=block_index,
                no_mask_labels=['text', 'formula', 'algorithm', 'reference', 'content', 'abstract'],
                is_eval=is_eval,
                is_only_xycut=is_only_xycut
            )
        self._recursive_img_array2path(self['layout_parsing_result'], save_path.parent,labels=['img'])
        super().save_to_markdown(save_path)

    def save_gt_json(self, gt_json_path, is_eval=True, is_only_xycut=False):
        """
        Save the ground truth data to a JSON file.

        Args:
            gt_json_path (str or Path): The path where the JSON file should be saved.
            is_eval (bool): Flag indicating whether to evaluate specific conditions.
            is_only_xycut (bool): Flag indicating whether to apply only XY cut.

        Returns:
            None
        """
        parsing_result = self['layout_parsing_result']
        input_data = [{
            'block_bbox': block['block_bbox'],
            'sub_indices': [],
            'sub_bboxes': [],
            'block_size': block['block_size'],
            'page_idx': None
        } for block in parsing_result]

        for block_index, block in enumerate(parsing_result):
            sub_blocks = block['sub_blocks']
            self._get_layout_ordering(
                block_index=block_index,
                no_mask_labels=['text', 'formula', 'algorithm', 'reference', 'content', 'abstract'],
                is_eval=is_eval,
                is_only_xycut=is_only_xycut
            )
            for sub_block in sub_blocks:
                input_data[block_index]["sub_bboxes"].append(list(map(int, sub_block["layout_bbox"])))
                input_data[block_index]["sub_indices"].append(int(sub_block["index"]))

        # Load existing data from the JSON file
        try:
            with open(gt_json_path, 'r', encoding='utf-8') as file:
                data = json.load(file) or []
        except Exception as e:
            data = []

        # Update page indices and extend the data
        start_idx = len(data)
        for block in input_data:
            block['page_idx'] = start_idx
            start_idx += 1
        data.extend(input_data)

        # Save updated data back to the JSON file
        with open(gt_json_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=4, ensure_ascii=False)
        

    def eval_layout_ordering(self, gt_json_path=None):
        """
        Evaluate the layout ordering by comparing the generated layout parsing result with ground truth data.

        :param gt_json_path: str, optional
            The file path to the ground truth JSON data. If not provided, this method assumes that the ground truth data
            is available through other means.

        :return: tuple
            A tuple containing the following evaluation metrics:
            - bleu_score: The BLEU score evaluating the similarity of the generated layout to the ground truth.
            - ard: The Average Relative Difference metric for layout comparison.
            - tau: Kendall's Tau, measuring the correlation between the generated layout order and the ground truth order.
        """
        gt_data = self._load_gt_data_from_json(gt_json_path)
        parsing_result = self['layout_parsing_result']
        input_data = self._generate_input_data(parsing_result)
        bleu_score, ard, tau = calculate_metrics_with_page(input_data, gt_data)
        # print(f"BLEU score: {bleu_score}, ARD: {ard}, TAU:{tau}")
        return bleu_score, ard, tau
    
    def _get_show_color(self,label):
        label_colors = {
        'paragraph_title': (102, 102, 255, 100),  # Medium Blue (from 'titles_list')
        'doc_title': (255, 248, 220, 100),        # Cornsilk
        'table_title': (255, 255, 102, 100),      # Light Yellow (from 'tables_caption_list')
        'figure_title': (102, 178, 255, 100),     # Sky Blue (from 'imgs_caption_list')
        'chart_title': (221, 160, 221, 100),      # Plum
        'vision_footnote': (144, 238, 144, 100),  # Light Green
        'text': (153, 0, 76, 100),                # Deep Purple (from 'texts_list')
        'formula': (0, 255, 0, 100),              # Bright Green (from 'interequations_list')
        'abstract': (255, 239, 213, 100),         # Papaya Whip
        'content': (40, 169, 92, 100),            # Medium Green (from 'lists_list' and 'indexs_list')
        'seal': (158, 158, 158, 100),             # Neutral Gray (from 'dropped_bbox_list')
        'table': (204, 204, 0, 100),              # Olive Yellow (from 'tables_body_list')
        'image': (153, 255, 51, 100),             # Bright Green (from 'imgs_body_list')
        'figure': (153, 255, 51, 100),             # Bright Green (from 'imgs_body_list')
        'chart': (216, 191, 216, 100),            # Thistle
        'reference': (229, 255, 204, 100),        # Pale Yellow-Green (from 'tables_footnote_list')
        'algorithm': (255, 250, 240, 100)         # Floral White
        }
        default_color = (158, 158, 158, 100)
        return label_colors.get(label, default_color)

    def _load_gt_data_from_json(self,gt_json_path):
        with open(gt_json_path, 'r', encoding='utf-8') as file:
            gt_data = json.load(file)
        return gt_data

    def _img_array2path(self,data,save_path):
        if isinstance(data, np.ndarray) and data.ndim == 3:  # Check if it's an image array
            # Generate a unique filename using UUID
            img_name = f"image_{uuid.uuid4().hex}.png"
            img_path = Path(save_path) / "imgs" / img_name
            img_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
            cv2.imwrite(str(img_path), data)
            return f"imgs/{img_name}"
        else:
            return ValueError

    def _recursive_img_array2path(self, data, save_path, labels=[]):
        """
        Process a dictionary or list to save image arrays to disk and replace them with file paths.

        Args:
            data (dict or list): The data structure that may contain image arrays.
            save_path (str or Path): The base path where images should be saved.
        """
        if isinstance(data, dict):
            for k, v in data.items():
                if k in labels and isinstance(v, np.ndarray) and v.ndim == 3:  # Check if it's an image array
                    data[k] = self._img_array2path(v, save_path)
                else:
                    self._recursive_img_array2path(v, save_path,labels)
        elif isinstance(data, list):
            for item in data:
                self._recursive_img_array2path(item, save_path,labels)

    def _calculate_overlap_area_2_minbox_area_ratio(self, bbox1, bbox2):
        """
        Calculate the ratio of the overlap area between bbox1 and bbox2 
        to the area of the smaller bounding box.

        Args:
            bbox1 (list or tuple): Coordinates of the first bounding box [x_min, y_min, x_max, y_max].
            bbox2 (list or tuple): Coordinates of the second bounding box [x_min, y_min, x_max, y_max].

        Returns:
            float: The ratio of the overlap area to the area of the smaller bounding box.
        """
        x_left = max(bbox1[0], bbox2[0])
        y_top = max(bbox1[1], bbox2[1])
        x_right = min(bbox1[2], bbox2[2])
        y_bottom = min(bbox1[3], bbox2[3])
        if x_right <= x_left or y_bottom <= y_top:
            return 0.0
        # Calculate the area of the overlap
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        # Calculate the areas of both bounding boxes
        area_bbox1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area_bbox2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        # Determine the minimum non-zero box area
        min_box_area = min(area_bbox1, area_bbox2)
        # Avoid division by zero in case of zero-area boxes
        if min_box_area == 0:
            return 0.0
        return intersection_area / min_box_area
        
    def _get_minbox_if_overlap_by_ratio(self,bbox1, bbox2, ratio, smaller=True):
        """
        Determine if the overlap area between two bounding boxes exceeds a given ratio 
        and return the smaller (or larger) bounding box based on the `smaller` flag.

        Args:
            bbox1 (list or tuple): Coordinates of the first bounding box [x_min, y_min, x_max, y_max].
            bbox2 (list or tuple): Coordinates of the second bounding box [x_min, y_min, x_max, y_max].
            ratio (float): The overlap ratio threshold.
            smaller (bool): If True, return the smaller bounding box; otherwise, return the larger one.

        Returns:
            list or tuple: The selected bounding box or None if the overlap ratio is not exceeded.
        """
        # Calculate the areas of both bounding boxes
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        # Calculate the overlap ratio using a helper function
        overlap_ratio = self._calculate_overlap_area_2_minbox_area_ratio(bbox1, bbox2)
        # Check if the overlap ratio exceeds the threshold
        if overlap_ratio > ratio:
            if (area1 <= area2 and smaller) or (area1 >= area2 and not smaller):
                return 1
            else:
                return 2
        return None


    def _remove_overlap_blocks(self,blocks,threshold=0.65,smaller=True):
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
        dropped_indexes = []
        # Iterate over each pair of blocks to find overlaps
        for i in range(len(blocks)):
            block1 = blocks[i]
            for j in range(i+1,len(blocks)):
                block2 = blocks[j]
                # Skip blocks that are already marked for removal
                if i in dropped_indexes or j in dropped_indexes:
                    continue
                # Check for overlap and determine which block to remove
                overlap_box_index = self._get_minbox_if_overlap_by_ratio(
                    block1['layout_bbox'], block2['layout_bbox'], threshold, smaller=smaller
                )
                if overlap_box_index is not None:
                    if overlap_box_index == 1:
                        block_to_remove = block1
                        drop_index = i
                    else:
                        block_to_remove = block2
                        drop_index = j
                    if drop_index not in dropped_indexes:
                        dropped_indexes.append(drop_index)
                        dropped_blocks.append(block_to_remove)
        
        dropped_indexes.sort()
        for i in reversed(dropped_indexes):
            del blocks[i]

        return blocks, dropped_blocks
    

    def _text_median_width(self, blocks):
        # widths = [block['layout_bbox'][2] - block['layout_bbox'][0] for block in blocks if block['label'] in ['text','formula','algorithm','reference','content',"doc_title","paragraph_title","abstract"]]
        widths = [block['layout_bbox'][2] - block['layout_bbox'][0] for block in blocks if block['label'] in ['text']]
        return np.median(widths) if widths else float('inf')

    def _get_layout_property(self, blocks, median_width, no_mask_labels, threshold=0.8):
        """
        Determine the layout (single or double column) of text blocks.

        Args:
            blocks (list): List of block dictionaries containing 'label' and 'layout_bbox'.
            median_width (float): Median width of text blocks.
            threshold (float): Threshold for determining layout overlap.

        Returns:
            list: Updated list of blocks with layout information.
        """
        blocks.sort(key=lambda x: (x['layout_bbox'][0], (x['layout_bbox'][2] - x['layout_bbox'][0])))
        check_single_layout = {}
        page_min_y,page_max_y = float('inf'), 0
        page_min_x,page_max_x = float('inf'), 0
        double_label_height = 0
        double_label_area = 0
        single_label_area = 0

        for i, block in enumerate(blocks):
            page_min_x = min(page_min_x, block['layout_bbox'][0])
            page_max_x = max(page_max_x, block['layout_bbox'][2])
            page_min_y = min(page_min_y, block['layout_bbox'][1])
            page_max_y = max(page_max_y, block['layout_bbox'][3])
        page_width = page_max_x - page_min_x
        page_height = page_max_y - page_min_y

        for i, block in enumerate(blocks):
            if block['label'] not in no_mask_labels:
                continue

            x_min_i, _, x_max_i, _ = block['layout_bbox']
            layout_length = x_max_i - x_min_i
            cover_count, cover_with_threshold_count = 0, 0
            match_block_with_threshold_indexes = []

            for j, other_block in enumerate(blocks):
                if i == j or other_block['label'] not in no_mask_labels:
                    continue

                x_min_j, _, x_max_j, _ = other_block['layout_bbox']
                x_match_min, x_match_max = max(x_min_i, x_min_j), min(x_max_i, x_max_j)
                match_block_iou = (x_match_max - x_match_min) / (x_max_j - x_min_j)

                if match_block_iou > 0:
                    cover_count += 1
                    if match_block_iou > threshold:
                        cover_with_threshold_count += 1
                        match_block_with_threshold_indexes.append((j, match_block_iou))
                    x_min_i = x_match_max
                    if x_min_i >= x_max_i:
                        break

            if (layout_length > median_width * 1.3 and (cover_with_threshold_count >= 2 or cover_count >= 2)) or \
                layout_length > 0.6 * page_width:
            # if layout_length > median_width * 1.3 and (cover_with_threshold_count >= 2):
                block['layout'] = "double"
                double_label_height += block['layout_bbox'][3] - block['layout_bbox'][1]
                double_label_area += (block['layout_bbox'][2] - block['layout_bbox'][0]) * \
                    (block['layout_bbox'][3] - block['layout_bbox'][1])
            else:
                block['layout'] = "single"
                check_single_layout[i] = match_block_with_threshold_indexes
        
        # Check single-layout block 
        for i, single_layout in check_single_layout.items():
            if single_layout:
                index, match_iou = single_layout[-1]
                if match_iou > 0.9 and blocks[index]['layout'] == "double":
                    blocks[i]['layout'] = "double"
                    double_label_height += blocks[i]['layout_bbox'][3] - blocks[i]['layout_bbox'][1]
                    double_label_area += (blocks[i]['layout_bbox'][2] - blocks[i]['layout_bbox'][0]) * \
                        (blocks[i]['layout_bbox'][3] - blocks[i]['layout_bbox'][1])
                else:
                    single_label_area += (blocks[i]['layout_bbox'][2] - blocks[i]['layout_bbox'][0]) * \
                        (blocks[i]['layout_bbox'][3] - blocks[i]['layout_bbox'][1])

        # return blocks,(double_label_height > 0.6 * page_height) or (double_label_area > single_label_area)
        return blocks, (double_label_area > single_label_area)
    
    def _get_bbox_direction(self, input_bbox, ratio=1):
        """
        Determine if a bounding box is horizontal or vertical.

        Args:
            input_bbox (list): Bounding box [x_min, y_min, x_max, y_max].
            ratio (float): Ratio for determining orientation.

        Returns:
            bool: True if horizontal, False if vertical.
        """
        return (input_bbox[2] - input_bbox[0]) * ratio >= (input_bbox[3] - input_bbox[1])

    def _get_projection_iou(self, input_bbox, match_bbox, is_horizontal=True):
        """
        Calculate the IoU of lines between two bounding boxes.

        Args:
            input_bbox (list): First bounding box [x_min, y_min, x_max, y_max].
            match_bbox (list): Second bounding box [x_min, y_min, x_max, y_max].
            is_horizontal (bool): Whether to compare horizontally or vertically.

        Returns:
            float: Line IoU.
        """
        if is_horizontal:
            x_match_min = max(input_bbox[0], match_bbox[0])
            x_match_max = min(input_bbox[2], match_bbox[2])
            return (x_match_max - x_match_min) / (input_bbox[2] - input_bbox[0])
        else:
            y_match_min = max(input_bbox[1], match_bbox[1])
            y_match_max = min(input_bbox[3], match_bbox[3])
            return (y_match_max - y_match_min) / (input_bbox[3] - input_bbox[1])

    def _get_sub_category(self, blocks, title_labels):
        """
        Determine the layout of title and text blocks.

        Args:
            blocks (list): List of block dictionaries.
            title_labels (list): List of labels considered as titles.

        Returns:
            list: Updated list of blocks with title-text layout information.
        """

        sub_title_labels = ['paragraph_title']
        vision_labels = ['image','table','chart','figure']

        for i, block1 in enumerate(blocks):
            if block1.get('title_text') is None:
                block1['title_text'] = []
            if block1.get('sub_title') is None:
                block1['sub_title'] = []
            if block1.get('vision_footnote') is None:
                block1['vision_footnote'] = []
            if block1.get("sub_label") is None:
                block1['sub_label'] = block1['label']

            if block1['label'] not in title_labels and block1['label'] not in sub_title_labels and block1['label'] not in vision_labels:
                continue
            
            bbox1 = block1['layout_bbox']
            x1, y1, x2, y2 = bbox1
            is_horizontal_1 = self._get_bbox_direction(block1['layout_bbox'])
            left_up_title_text_distance = float('inf')
            left_up_title_text_index = -1
            left_up_title_text_direction = None
            right_down_title_text_distance = float('inf')
            right_down_title_text_index = -1
            right_down_title_text_direction = None

            for j, block2 in enumerate(blocks):
                if i == j:
                    continue
                
                bbox2 = block2['layout_bbox']
                x1_prime, y1_prime, x2_prime, y2_prime = bbox2
                is_horizontal_2 = self._get_bbox_direction(bbox2)
                match_block_iou = self._get_projection_iou(bbox2, bbox1, is_horizontal_1)

                def distance_(is_horizontal,is_left_up):
                    if is_horizontal:
                        if is_left_up:
                            return (y1 - y2_prime + 2) // 5 + x1_prime / 5000
                        else:
                            return (y1_prime - y2 + 2) // 5 + x1_prime / 5000
                        
                    else:
                        if is_left_up:
                            return (x1 - x2_prime + 2) // 5 + y1_prime / 5000
                        else:
                            return (x1_prime - x2 + 2) // 5 + y1_prime / 5000

                block_iou_threshold = 0.1
                if block1['label'] in sub_title_labels:
                    match_block_iou = self._calculate_overlap_area_2_minbox_area_ratio(bbox2, bbox1)
                    block_iou_threshold = 0.7

                if is_horizontal_1:
                    if match_block_iou >= block_iou_threshold:
                        left_up_distance = distance_(True,True)
                        right_down_distance = distance_(True,False)
                        if y2_prime <= y1 and left_up_distance <= left_up_title_text_distance:
                            left_up_title_text_distance = left_up_distance
                            left_up_title_text_index = j
                            left_up_title_text_direction = is_horizontal_2
                        elif y1_prime > y2 and right_down_distance < right_down_title_text_distance:
                            right_down_title_text_distance = right_down_distance
                            right_down_title_text_index = j
                            right_down_title_text_direction = is_horizontal_2
                else:
                    if match_block_iou >= block_iou_threshold:
                        left_up_distance = distance_(False,True)
                        right_down_distance = distance_(False,False)
                        if x2_prime <= x1 and left_up_distance <= left_up_title_text_distance:
                            left_up_title_text_distance = left_up_distance
                            left_up_title_text_index = j
                            left_up_title_text_direction = is_horizontal_2
                        elif x1_prime > x2 and right_down_distance < right_down_title_text_distance:
                            right_down_title_text_distance = right_down_distance
                            right_down_title_text_index = j
                            right_down_title_text_direction = is_horizontal_2

            height =bbox1[3] - bbox1[1]
            width = bbox1[2] - bbox1[0]
            title_text_weight = [0.8, 0.8]
            # title_text_weight = [2, 2]

            title_text = []
            sub_title = []
            vision_footnote = []
            
            def get_sub_category_(title_text_direction,title_text_index,label,is_left_up=True):
                direction_ = [1,3] if is_left_up else [2,4]
                if title_text_direction == is_horizontal_1 and title_text_index != -1 and (label == "text" or label == "paragraph_title"):
                    bbox2 = blocks[title_text_index]['layout_bbox']
                    if is_horizontal_1:
                        height1 = bbox2[3] - bbox2[1]
                        width1 = bbox2[2] - bbox2[0]
                        if label == "text" :
                            if self._nearest_edge_distance(bbox1,bbox2)[0] <= 15 and block1['label'] in vision_labels and \
                                    width1 < width and height1 < 0.5*height:
                                blocks[title_text_index]['sub_label'] = 'vision_footnote'
                                vision_footnote.append(bbox2)
                            elif height1 < height * title_text_weight[0] and (width1 < width or width1 > 1.5 * width) and block1['label'] in title_labels:
                                blocks[title_text_index]['sub_label'] = "title_text"
                                title_text.append((direction_[0], bbox2))
                        elif label == "paragraph_title" and block1['label'] in  sub_title_labels:
                            sub_title.append(bbox2)
                    else:
                        height1 = bbox2[3] - bbox2[1]
                        width1 = bbox2[2] - bbox2[0]
                        if label == "text":    
                            if self._nearest_edge_distance(bbox1,bbox2)[0] <= 15 and block1['label'] in vision_labels and \
                                    height1 < height and width1 < 0.5*width:
                                blocks[title_text_index]['sub_label'] = 'vision_footnote'
                                vision_footnote.append(bbox2)
                            elif width1 < width * title_text_weight[1] and block1['label'] in title_labels:
                                blocks[title_text_index]['sub_label'] = "title_text"
                                title_text.append((direction_[1], bbox2))
                        elif label == "paragraph_title" and block1['label'] in  sub_title_labels:
                            sub_title.append(bbox2)

            if (is_horizontal_1 and abs(left_up_title_text_distance - right_down_title_text_distance)*5 > height) or \
                (not is_horizontal_1 and abs(left_up_title_text_distance - right_down_title_text_distance)*5 > width) :
                if left_up_title_text_distance < right_down_title_text_distance:
                    get_sub_category_(left_up_title_text_direction,
                                    left_up_title_text_index,
                                    blocks[left_up_title_text_index]['label'],True)
                else:
                    get_sub_category_(right_down_title_text_direction,
                                    right_down_title_text_index,
                                    blocks[right_down_title_text_index]['label'],False)
            else:
                get_sub_category_(left_up_title_text_direction,
                                left_up_title_text_index,
                                blocks[left_up_title_text_index]['label'],True)
                get_sub_category_(right_down_title_text_direction,
                                right_down_title_text_index,
                                blocks[right_down_title_text_index]['label'],False)

            if block1['label'] in title_labels:
                if blocks[i].get('title_text') == []:
                    blocks[i]['title_text'] = title_text
            
            if block1['label'] in sub_title_labels:
                if blocks[i].get('sub_title') == []:
                    blocks[i]['sub_title'] = sub_title
                
            if block1['label'] in vision_labels:
                if blocks[i].get('vision_footnote') == []:
                    blocks[i]['vision_footnote'] = vision_footnote

        return blocks     


    def _get_layout_ordering(self,block_index=0,no_mask_labels=[],is_eval=False,is_only_xycut=False):
        """
        Process layout parsing results to remove overlapping bounding boxes 
        and assign an ordering index based on their positions.

        Modifies:
            The 'parsing_result' list in 'layout_parsing_result' by adding an 'index' to each block.

        """
        if self.is_ordered:
            return 
        title_text_labels = ["doc_title"]
        title_labels = ["doc_title", "paragraph_title"]
        vision_labels = ['image','table','seal','chart','figure']
        vision_title_labels = ["table_title", 'chart_title', "figure_title"]

        layout_parsing_result = self['layout_parsing_result']
        parsing_result = layout_parsing_result[block_index]['sub_blocks']
        parsing_result, _ = self._remove_overlap_blocks(parsing_result, threshold=0.5, smaller=True)
        parsing_result = self._get_sub_category(parsing_result,title_text_labels)

        if is_only_xycut == False:
            # title_labels = ["doc_title","paragraph_title"]
            doc_flag = False
            median_width = self._text_median_width(parsing_result)
            parsing_result,projection_direction = self._get_layout_property(parsing_result,
                                                                            median_width,
                                                                            no_mask_labels=no_mask_labels, 
                                                                            threshold=0.3)
            # Convert bounding boxes to float and remove overlaps
            double_text_blocks, title_text_blocks, title_blocks, vision_blocks, vision_title_blocks, vision_footnote_blocks, other_blocks = [], [], [], [], [], [], []
            
            drop_indexes = []

            for index,block in enumerate(parsing_result):
                label = block['sub_label']
                block['layout_bbox'] = list(map(int, block['layout_bbox']))
                
                if label == "doc_title":
                    doc_flag = True

                if label in no_mask_labels:
                    if block["layout"] == "double":
                        double_text_blocks.append(block)
                        drop_indexes.append(index) 
                elif label == "title_text":
                    title_text_blocks.append(block)
                    drop_indexes.append(index)     
                elif label == "vision_footnote":
                    vision_footnote_blocks.append(block)
                    drop_indexes.append(index) 
                elif label in vision_title_labels:
                    vision_title_blocks.append(block)
                    drop_indexes.append(index)
                elif label in title_labels:
                    title_blocks.append(block)
                    drop_indexes.append(index) 
                elif label in vision_labels:
                    vision_blocks.append(block)
                    drop_indexes.append(index) 
                else:
                    other_blocks.append(block)
                    drop_indexes.append(index)

            for index in sorted(drop_indexes, reverse=True):
                del parsing_result[index]

        if len(parsing_result)>0:
            # single text label
            if is_only_xycut==False and \
                (len(double_text_blocks) > len(parsing_result) or projection_direction):
                parsing_result.extend(title_blocks + double_text_blocks)
                title_blocks = []
                double_text_blocks = []
                block_bboxes = [block["layout_bbox"] for block in parsing_result]
                block_bboxes.sort(key=lambda x: (x[0]//max(20,median_width),x[1]))
                block_bboxes = np.array(block_bboxes)
                print("sort by yxcut...")
                sorted_indices = sort_by_xycut(block_bboxes,
                                               direction=1,
                                               min_gap=1)    
            else:
                block_bboxes = [block["layout_bbox"] for block in parsing_result]
                block_bboxes.sort(key=lambda x: (x[0]//20,x[1]))
                block_bboxes = np.array(block_bboxes)
                sorted_indices = sort_by_xycut(block_bboxes,
                                               direction=0,
                                               min_gap=20)
                
            sorted_boxes = block_bboxes[sorted_indices].tolist()

            for block in parsing_result:
                block['index'] = sorted_boxes.index(block['layout_bbox'])+1
                block['sub_index'] = sorted_boxes.index(block['layout_bbox'])+1 

        def nearest_match_(input_blocks,distance_type="manhattan",is_add_index=True):
            for block in input_blocks:
                bbox = block['layout_bbox']
                min_distance = float('inf')
                min_distance_config = [[float('inf'),float('inf')],float('inf'),float('inf')] # for double text
                nearest_gt_index = 0
                for match_block in parsing_result:
                    match_bbox = match_block['layout_bbox']
                    if distance_type=="nearest_iou_edge_distance":
                        distance,min_distance_config = self._nearest_iou_edge_distance(bbox, 
                                                                match_bbox,
                                                                block['sub_label'],
                                                                vision_labels=vision_labels,
                                                                no_mask_labels=no_mask_labels,
                                                                median_width=median_width,
                                                                title_labels = title_labels,
                                                                title_text=block['title_text'],
                                                                sub_title=block['sub_title'],
                                                                min_distance_config=min_distance_config,
                                                                tolerance_len=10)
                    elif distance_type=="title_text":
                        if match_block['label'] in title_labels+["abstract"] and match_block['title_text'] != []:
                            iou_left_up = self._calculate_overlap_area_2_minbox_area_ratio(bbox,match_block['title_text'][0][1])
                            iou_right_down = self._calculate_overlap_area_2_minbox_area_ratio(bbox,match_block['title_text'][-1][1])
                            iou = 1-max(iou_left_up,iou_right_down)
                            distance = self._manhattan_distance(bbox, match_bbox)*iou
                        else:
                            distance = float('inf')
                    elif distance_type=="manhattan":
                        distance = self._manhattan_distance(bbox, match_bbox)
                    elif distance_type=="vision_footnote":
                        if match_block['label'] in vision_labels and match_block['vision_footnote'] != []:
                            iou_left_up = self._calculate_overlap_area_2_minbox_area_ratio(bbox,match_block['vision_footnote'][0])
                            iou_right_down = self._calculate_overlap_area_2_minbox_area_ratio(bbox,match_block['vision_footnote'][-1])
                            iou = 1-max(iou_left_up,iou_right_down)
                            distance = self._manhattan_distance(bbox, match_bbox)*iou
                        else:
                            distance = float('inf')
                    elif distance_type=="vision_body":
                        if match_block['label'] in vision_title_labels and block['vision_footnote'] != []:
                            iou_left_up = self._calculate_overlap_area_2_minbox_area_ratio(match_bbox,block['vision_footnote'][0])
                            iou_right_down = self._calculate_overlap_area_2_minbox_area_ratio(match_bbox,block['vision_footnote'][-1])
                            iou = 1-max(iou_left_up,iou_right_down)
                            distance = self._manhattan_distance(bbox, match_bbox)*iou
                        else:
                            distance = float('inf')
                    else:
                        raise NotImplementedError
                    
                    if distance < min_distance:
                        min_distance = distance
                        if is_add_index:
                            nearest_gt_index = match_block.get('index',999)
                        else:
                            nearest_gt_index = match_block.get('sub_index',999) 
                        
                if is_add_index:
                    block['index'] = nearest_gt_index
                else:
                    block['sub_index'] = nearest_gt_index

                parsing_result.append(block)
                    
            # for block in input_blocks:
            #     parsing_result.append(block)
        
        if is_only_xycut == False:

            # double text label
            double_text_blocks.sort(key=lambda x: (x['layout_bbox'][1]//10,x['layout_bbox'][0]//median_width,x['layout_bbox'][1]**2+x['layout_bbox'][0]**2))
            nearest_match_(double_text_blocks,distance_type="nearest_iou_edge_distance")
            parsing_result.sort(key=lambda x: (x['index'],x['layout_bbox'][1],x['layout_bbox'][0])) 

            for idx,block in enumerate(parsing_result):
                block['index'] = idx+1
                block['sub_index'] = idx+1

            # title label
            title_blocks.sort(key=lambda x: (x['layout_bbox'][1]//10,x['layout_bbox'][0]//median_width,x['layout_bbox'][1]**2+x['layout_bbox'][0]**2))
            nearest_match_(title_blocks,distance_type="nearest_iou_edge_distance")
            
            if doc_flag:
                # text_sort_labels = ["doc_title","paragraph_title","abstract"]
                text_sort_labels = ["doc_title"]
                text_label_priority = {label: priority for priority, label in enumerate(text_sort_labels)}
                doc_titles = []
                for i,block in enumerate(parsing_result):
                    if block['label'] == "doc_title":
                        doc_titles.append((i,block['layout_bbox'][1],block['layout_bbox'][0]))
                doc_titles.sort(key=lambda x: (x[1],x[2]))
                first_doc_title_index = doc_titles[0][0]
                parsing_result[first_doc_title_index]['index'] = 1
                parsing_result.sort(key=lambda x: (x['index'], text_label_priority.get(x['label'],9999),x['layout_bbox'][1],x['layout_bbox'][0])) 
            else:
                parsing_result.sort(key=lambda x: (x['index'],x['layout_bbox'][1],x['layout_bbox'][0])) 

            for idx,block in enumerate(parsing_result):
                block['index'] = idx+1
                block['sub_index'] = idx+1

            # title-text label
            nearest_match_(title_text_blocks,distance_type="title_text")
            text_sort_labels = ["doc_title","paragraph_title","title_text"]
            text_label_priority = {label: priority for priority, label in enumerate(text_sort_labels)}
            parsing_result.sort(key=lambda x: (x['index'],text_label_priority.get(x['sub_label'],9999),x['layout_bbox'][1],x['layout_bbox'][0])) 

            for idx,block in enumerate(parsing_result):
                block['index'] = idx+1
                block['sub_index'] = idx+1

            if is_eval == False:
                # image,figure,chart,seal label
                nearest_match_(vision_title_blocks,distance_type="nearest_iou_edge_distance",is_add_index=False)
                parsing_result.sort(key=lambda x: (x['sub_index'],x['layout_bbox'][1],x['layout_bbox'][0])) 

                for idx,block in enumerate(parsing_result):
                    block['sub_index'] = idx+1

                # image,figure,chart,seal label
                nearest_match_(vision_blocks,distance_type="nearest_iou_edge_distance",is_add_index=False)
                parsing_result.sort(key=lambda x: (x['sub_index'],x['layout_bbox'][1],x['layout_bbox'][0])) 

                for idx,block in enumerate(parsing_result):
                    block['sub_index'] = idx+1
                
                # vision footnote label
                nearest_match_(vision_footnote_blocks,distance_type="vision_footnote",is_add_index=False)
                text_label_priority = {'vision_footnote':9999}
                parsing_result.sort(key=lambda x: (x['sub_index'],text_label_priority.get(x['sub_label'],0),x['layout_bbox'][1],x['layout_bbox'][0])) 

                for idx,block in enumerate(parsing_result):
                    block['sub_index'] = idx+1
                
                # header、footnote、header_image... label 
                nearest_match_(other_blocks,distance_type="manhattan",is_add_index=False)
        
        self.is_ordered = True
    
    def _generate_input_data(self,parsing_result,is_eval=True,is_only_xycut=False):
        """
        The evaluation input data is generated based on the parsing results.

        :param parsing_result: A list containing the results of the layout parsing
        :return: A formatted list of input data
        """
        input_data = [{
            'block_bbox': block['block_bbox'],
            'sub_indices': [],
            'sub_bboxes': []
        } for block in parsing_result]
        
        for block_index, block in enumerate(parsing_result):
            sub_blocks = block['sub_blocks']
            self._get_layout_ordering(
                block_index=block_index,
                no_mask_labels=['text', 'formula', 'algorithm', 'reference', 'content', 'abstract'],
                is_eval=is_eval,
                is_only_xycut=is_only_xycut
            )
            for sub_block in sub_blocks:
                input_data[block_index]["sub_bboxes"].append(list(map(int, sub_block["layout_bbox"])))
                input_data[block_index]["sub_indices"].append(int(sub_block["index"]))

        return input_data

    def _manhattan_distance(self,point1, point2, weight_x=1, weight_y=1):
        return weight_x * abs(point1[0] - point2[0]) + weight_y * abs(point1[1] - point2[1])

    def calculate_horizontal_distance_(self, input_bbox, match_bbox, height, disperse, title_text):
        """
        Calculate the horizontal distance between two bounding boxes, considering title text adjustments.

        Args:
            input_bbox (list): The bounding box coordinates [x1, y1, x2, y2] of the input object.
            match_bbox (list): The bounding box coordinates [x1', y1', x2', y2'] of the object to match against.
            height (int): The height of the input bounding box used for normalization.
            disperse (int): The dispersion factor used to normalize the horizontal distance.
            title_text (list): A list of tuples containing title text information and their bounding box coordinates.
                            Format: [(position_indicator, [x1, y1, x2, y2]), ...].

        Returns:
            float: The calculated horizontal distance taking into account the title text adjustments.
        """
        x1, y1, x2, y2 = input_bbox
        x1_prime, y1_prime, x2_prime, y2_prime = match_bbox

        if y2 < y1_prime:
            if title_text and title_text[-1][0] == 2:
                y2 += title_text[-1][1][3] - title_text[-1][1][1]
            distance1 = (y1_prime - y2) * 0.5
        else:
            if title_text and title_text[0][0] == 1:
                y1 -= title_text[0][1][3] - title_text[0][1][1]
            distance1 = (y1 - y2_prime)

        return abs(x2_prime - x1) // disperse + distance1 // height + distance1 / 5000  # if page max size == 5000

    def calculate_vertical_distance_(self, input_bbox, match_bbox, width, disperse, title_text):
        """
        Calculate the vertical distance between two bounding boxes, considering title text adjustments.

        Args:
            input_bbox (list): The bounding box coordinates [x1, y1, x2, y2] of the input object.
            match_bbox (list): The bounding box coordinates [x1', y1', x2', y2'] of the object to match against.
            width (int): The width of the input bounding box used for normalization.
            disperse (int): The dispersion factor used to normalize the vertical distance.
            title_text (list): A list of tuples containing title text information and their bounding box coordinates.
                            Format: [(position_indicator, [x1, y1, x2, y2]), ...].

        Returns:
            float: The calculated vertical distance taking into account the title text adjustments.
        """
        x1, y1, x2, y2 = input_bbox
        x1_prime, y1_prime, x2_prime, y2_prime = match_bbox

        if x1 > x2_prime:
            if title_text and title_text[0][0] == 3:
                x1 -= title_text[0][1][2] - title_text[0][1][0]
            distance2 = (x1 - x2_prime) * 0.5
        else:
            if title_text and title_text[-1][0] == 4:
                x2 += title_text[-1][1][2] - title_text[-1][1][0]
            distance2 = (x1_prime - x2)

        return abs(y2_prime - y1) // disperse + distance2 // width + distance2 / 5000

    def _nearest_edge_distance(self, 
                               input_bbox, match_bbox, 
                               weight=[1, 1, 1, 1], 
                               label='text', no_mask_labels=[], 
                               min_edge_distances_config=[],tolerance_len = 10):
        """
        Calculate the nearest edge distance between two bounding boxes, considering directional weights.

        Args:
            input_bbox (list): The bounding box coordinates [x1, y1, x2, y2] of the input object.
            match_bbox (list): The bounding box coordinates [x1', y1', x2', y2'] of the object to match against.
            weight (list, optional): Directional weights for the edge distances [left, right, up, down]. Defaults to [1, 1, 1, 1].
            label (str, optional): The label/type of the object in the bounding box (e.g., 'text'). Defaults to 'text'.
            no_mask_labels (list, optional): Labels for which no masking is applied when calculating edge distances. Defaults to an empty list.
            min_edge_distances_config (list, optional): Configuration for minimum edge distances [min_edge_distance_x, min_edge_distance_y]. Defaults to [float('inf'), float('inf')].

        Returns:
            tuple: A tuple containing:
                - The calculated minimum edge distance between the bounding boxes.
                - A list with the minimum edge distances in the x and y directions.
        """
        match_bbox_iou = self._calculate_overlap_area_2_minbox_area_ratio(input_bbox, match_bbox)
        if match_bbox_iou > 0 and label not in no_mask_labels:
            return 0, [0, 0]
        
        if not min_edge_distances_config:
            min_edge_distances_config = [float('inf'), float('inf')]
        min_edge_distance_x, min_edge_distance_y = min_edge_distances_config
        
        x1, y1, x2, y2 = input_bbox
        x1_prime, y1_prime, x2_prime, y2_prime = match_bbox

        direction_num = 0
        distance_x = float('inf')
        distance_y = float('inf')
        distance = [float('inf')] * 4

        # input_bbox is to the left of match_bbox
        if x2 < x1_prime:
            direction_num += 1
            distance[0] = (x1_prime - x2)
            if abs(distance[0] - min_edge_distance_x) <= tolerance_len:
                distance_x = min_edge_distance_x * weight[0]
            else:
                distance_x = distance[0] * weight[0]
        # input_bbox is to the right of match_bbox
        elif x1 > x2_prime:
            direction_num += 1
            distance[1] = (x1 - x2_prime)
            if abs(distance[1] - min_edge_distance_x) <= tolerance_len:
                distance_x = min_edge_distance_x * weight[1]
            else:
                distance_x = distance[1] * weight[1]
        elif match_bbox_iou > 0:
            distance[0] = 0
            distance_x = 0

        # input_bbox is above match_bbox
        if y2 < y1_prime:
            direction_num += 1
            distance[2] = (y1_prime - y2)
            if abs(distance[2] - min_edge_distance_y) <= tolerance_len:
                distance_y = min_edge_distance_y * weight[2]
            else:
                distance_y = distance[2] * weight[2]
            if label in no_mask_labels:
                distance_y = max(0.1, distance_y) * 100
        # input_bbox is below match_bbox
        elif y1 > y2_prime:
            direction_num += 1
            distance[3] = (y1 - y2_prime)
            if abs(distance[3] - min_edge_distance_y) <= tolerance_len:
                distance_y = min_edge_distance_y * weight[3]
            else:
                distance_y = distance[3] * weight[3]
        elif match_bbox_iou > 0:
            distance[2] = 0
            distance_y = 0

        if direction_num == 2:
            return (distance_x + distance_y), \
                    [min(distance[0], distance[1]), min(distance[2], distance[3])]
        else:
            return min(distance_x, distance_y), \
                    [min(distance[0], distance[1]), min(distance[2], distance[3])]
    
    def _get_weights(self, label, horizontal):
        """Define weights based on the label and orientation."""
        if label == "doc_title":
            return [1, 0.1, 0.1, 1] if horizontal else [0.2, 0.1, 1, 1] # left-down ,  right-left 
        elif label in ["paragraph_title", "abstract", "figure_title", "chart_title", 'image','seal','chart','figure']:
            return [1, 1, 0.1, 1] # down 
        else:
            return [1, 1, 1, 0.1] # up

    def _nearest_iou_edge_distance(self, 
                                   input_bbox, match_bbox, 
                                   label, vision_labels, no_mask_labels, 
                                   median_width=-1, 
                                   title_labels=[], title_text=[], sub_title=[], 
                                   min_distance_config=[],
                                   tolerance_len =10):
        """
        Calculate the nearest IOU edge distance between two bounding boxes.

        Args:
            input_bbox (list): The bounding box coordinates [x1, y1, x2, y2] of the input object.
            match_bbox (list): The bounding box coordinates [x1', y1', x2', y2'] of the object to match against.
            label (str): The label/type of the object in the bounding box (e.g., 'image', 'text', etc.).
            no_mask_labels (list): Labels for which no masking is applied when calculating edge distances.
            median_width (int, optional): The median width for title dispersion calculation. Defaults to -1.
            title_labels (list, optional): Labels that indicate the object is a title. Defaults to an empty list.
            title_text (list, optional): Text content associated with title labels. Defaults to an empty list.
            sub_title (list, optional): List of subtitle bounding boxes to adjust the input_bbox. Defaults to an empty list.
            min_distance_config (list, optional): Configuration for minimum distances [min_edge_distances_config, up_edge_distances_config, total_distance].

        Returns:
            tuple: A tuple containing the calculated distance and updated minimum distance configuration.
        """

        x1, y1, x2, y2 = input_bbox
        x1_prime, y1_prime, x2_prime, y2_prime = match_bbox

        min_edge_distances_config, up_edge_distances_config, total_distance = min_distance_config

        iou_distance = 0

        if label in vision_labels:
            horizontal1 = horizontal2 = True
        else:
            horizontal1 = self._get_bbox_direction(input_bbox)
            horizontal2 = self._get_bbox_direction(match_bbox, 3)

        if horizontal1 != horizontal2 or self._get_projection_iou(input_bbox, match_bbox, horizontal1) < 0.01:
            iou_distance = 1
        elif label == "doc_title" or (label in title_labels and title_text):
            # Calculate distance for titles
            disperse = max(1, median_width)
            width = x2 - x1
            height = y2 - y1
            if horizontal1:
                return self.calculate_horizontal_distance_(input_bbox, match_bbox, height, disperse, title_text), \
                       min_distance_config
            else:
                return self.calculate_vertical_distance_(input_bbox, match_bbox, width, disperse, title_text), \
                       min_distance_config

        # Adjust input_bbox based on sub_title
        if sub_title:
            for sub in sub_title:
                x1_, y1_, x2_, y2_ = sub
                x1, y1, x2, y2 = min(x1, x1_), min(y1, y1_), max(x2, x2_), max(y2, y2_)
            input_bbox = [x1, y1, x2, y2]

        # Calculate edge distance
        weight = self._get_weights(label, horizontal1)
        if label == "abstract":
            tolerance_len *= 3
        edge_distance, edge_distance_config = self._nearest_edge_distance(input_bbox, match_bbox, weight, 
                                                                        label=label, 
                                                                        no_mask_labels=no_mask_labels, 
                                                                        min_edge_distances_config=min_edge_distances_config,
                                                                        tolerance_len=tolerance_len)

        # Weights for combining distances
        iou_edge_weight = [10**6, 10**3, 1, 0.001]

        # Calculate up and left edge distances
        up_edge_distance = y1_prime
        left_edge_distance = x1_prime
        if (label in no_mask_labels or label == "paragraph_title" or label in vision_labels) and y1 > y2_prime:
            up_edge_distance = -y2_prime
            left_edge_distance = -x2_prime

        min_up_edge_distance = up_edge_distances_config
        if abs(min_up_edge_distance - up_edge_distance) <= tolerance_len:
            up_edge_distance = min_up_edge_distance

        # Calculate total distance
        distance = (
            iou_distance * iou_edge_weight[0] +
            edge_distance * iou_edge_weight[1] +
            up_edge_distance * iou_edge_weight[2] +
            left_edge_distance * iou_edge_weight[3]
        )

        # Update minimum distance configuration if a smaller distance is found
        if total_distance > distance:
            edge_distance_config = [
                min(min_edge_distances_config[0], edge_distance_config[0]),
                min(min_edge_distances_config[1], edge_distance_config[1])
            ]
            min_distance_config = [edge_distance_config, min(up_edge_distance, up_edge_distances_config), distance]

        return distance, min_distance_config