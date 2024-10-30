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
import os.path as osp
from collections import defaultdict, Counter
from pathlib import Path
import PIL
from PIL import Image, ImageOps, ImageDraw, ImageFont
import json
from pycocotools.coco import COCO
import numpy as np
from .....utils.errors import DatasetFileNotFoundError
from .utils.visualizer import draw_bbox
from .....utils.fonts import PINGFANG_FONT_FILE_PATH


def check(dataset_dir, output, model_name):
    """check dataset"""
    dataset_dir = osp.abspath(dataset_dir)
    if not osp.exists(dataset_dir) or not osp.isdir(dataset_dir):
        raise DatasetFileNotFoundError(file_path=dataset_dir)

    sample_cnts = dict()
    sample_paths = defaultdict(list)
    im_sizes = defaultdict(Counter)
    if model_name == 'FairMOT-DLA-34':
        num_class = 1
        tags = {'train':['mot17.train'], 'val':['mot17.train']}
        for tag in tags.keys():
            default_image_lists = tags[tag]
            img_files = {}
            samp_num = 0
            for data_name in default_image_lists:
                list_path = osp.join(dataset_dir, 'image_lists', data_name)
                with open(list_path, 'r') as file:
                    img_files[data_name] = file.readlines()
                    img_files[data_name] = [
                        os.path.join(dataset_dir, x.strip())
                        for x in img_files[data_name]
                    ]
                    img_files[data_name] = list(
                        filter(lambda x: len(x) > 0, img_files[data_name]))

            image_info = []
            for data_name in img_files.keys():
                samp_num += len(img_files[data_name])
                image_info += img_files[data_name]
            
            label_files_info = [
                x.replace('images', 'labels_with_ids').replace(
                    '.png', '.txt').replace('.jpg', '.txt')
                for x in image_info
            ]

            sample_num = min(10, samp_num)
            
            for i in range(sample_num):
                img_path = image_info[i]
                label_path = label_files_info[i]
                labels = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 6)  # [gt_class, gt_identity, cx, cy, w, h]
                
                if not osp.exists(img_path):
                    raise DatasetFileNotFoundError(file_path=img_path)
                img = Image.open(img_path)

                labels[:, [2, 4]] *= img.width
                labels[:, [3, 5]] *= img.height

                img = ImageOps.exif_transpose(img)
                vis_im = draw_bbox_with_labels(img, labels)
                vis_save_dir = osp.join(output, "demo_img", tag)
                vis_path = osp.join(vis_save_dir, str(i)+'.'+img_path.split('.')[1])
                Path(vis_path).parent.mkdir(parents=True, exist_ok=True)
                vis_im.save(vis_path)
                sample_path = osp.join(
                    "check_dataset", os.path.relpath(vis_path, output)
                )
                sample_paths[tag].append(sample_path)
        
        attrs = {}
        attrs["num_classes"] = num_class
        attrs["train_samples"] = samp_num
        attrs["train_sample_paths"] = sample_paths['train']

        attrs["val_samples"] = len(img_files['mot17.train'])
        attrs["val_sample_paths"] = sample_paths['val']

    else:
        tags = ["train_half", "val_half"]
        for _, tag in enumerate(tags):
            file_list = osp.join(dataset_dir, f"annotations/{tag}.json")
            if not osp.exists(file_list):
                if tag in ("train_half", "val_half"):
                    # train and val file lists must exist
                    raise DatasetFileNotFoundError(
                        file_path=file_list,
                        solution=f"Ensure that both `train_half.json` and `val_half.json` exist in \
                                    {dataset_dir}/annotations",
                    )
                else:
                    continue
            else:
                with open(file_list, "r", encoding="utf-8") as f:
                    jsondata = json.load(f)

                coco = COCO(file_list)
                num_class = len(coco.getCatIds())

                vis_save_dir = osp.join(output, "demo_img")

                image_info = jsondata["images"]
                sample_cnts[tag] = len(image_info)
                sample_num = min(10, len(image_info))
                img_types = {"train_half": 'train', "val_half": 'val'}
                for i in range(sample_num):
                    file_name = image_info[i]["file_name"]
                    img_id = image_info[i]["id"]
                    img_type = img_types[tag]
                    img_path = osp.join(dataset_dir, "images", 'train', file_name)
                    if not osp.exists(img_path):
                        raise DatasetFileNotFoundError(file_path=img_path)
                    img = Image.open(img_path)
                    img = ImageOps.exif_transpose(img)
                    vis_im = draw_bbox(img, coco, img_id)
                    vis_path = osp.join(vis_save_dir, file_name)
                    Path(vis_path).parent.mkdir(parents=True, exist_ok=True)
                    vis_im.save(vis_path)
                    sample_path = osp.join(
                        "check_dataset", os.path.relpath(vis_path, output)
                    )
                    sample_paths[tag].append(sample_path)

        attrs = {}
        attrs["num_classes"] = num_class
        attrs["train_samples"] = sample_cnts["train_half"]
        attrs["train_sample_paths"] = sample_paths["train_half"]

        attrs["val_samples"] = sample_cnts["val_half"]
        attrs["val_sample_paths"] = sample_paths["val_half"]
    return attrs

def font_colormap(color_index):
    """
    Get font color according to the index of colormap
    """
    dark = np.array([0x14, 0x0E, 0x35])
    light = np.array([0xFF, 0xFF, 0xFF])
    light_indexs = [0, 3, 4, 8, 9, 13, 14, 18, 19]
    if color_index in light_indexs:
        return light.astype("int32")
    else:
        return dark.astype("int32")

def colormap(rgb=False):
    """
    Get colormap

    The code of this function is copied from https://github.com/facebookresearch/Detectron/blob/main/detectron/\
    utils/colormap.py
    """
    color_list = np.array(
        [
            0xFF,
            0x00,
            0x00,
            0xCC,
            0xFF,
            0x00,
            0x00,
            0xFF,
            0x66,
            0x00,
            0x66,
            0xFF,
            0xCC,
            0x00,
            0xFF,
            0xFF,
            0x4D,
            0x00,
            0x80,
            0xFF,
            0x00,
            0x00,
            0xFF,
            0xB2,
            0x00,
            0x1A,
            0xFF,
            0xFF,
            0x00,
            0xE5,
            0xFF,
            0x99,
            0x00,
            0x33,
            0xFF,
            0x00,
            0x00,
            0xFF,
            0xFF,
            0x33,
            0x00,
            0xFF,
            0xFF,
            0x00,
            0x99,
            0xFF,
            0xE5,
            0x00,
            0x00,
            0xFF,
            0x1A,
            0x00,
            0xB2,
            0xFF,
            0x80,
            0x00,
            0xFF,
            0xFF,
            0x00,
            0x4D,
        ]
    ).astype(np.float32)
    color_list = color_list.reshape((-1, 3))
    if not rgb:
        color_list = color_list[:, ::-1]
    return color_list.astype("int32")

def draw_bbox_with_labels(image, label):
    """
    Draw bbox on image
    """
    font_size = 12
    font = ImageFont.truetype(PINGFANG_FONT_FILE_PATH, font_size, encoding="utf-8")

    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)
    image_size = image.size
    width = int(max(image_size) * 0.005)

    catid2color = {}
    catid2fontcolor = {}
    catid_num_dict = {}
    color_list = colormap(rgb=True)
    annotations = label

    for ann in annotations:
        catid = int(ann[1])
        catid_num_dict[catid] = catid_num_dict.get(catid, 0) + 1
    for i, (catid, _) in enumerate(
        sorted(catid_num_dict.items(), key=lambda x: x[1], reverse=True)
    ):
        if catid not in catid2color:
            color_index = i % len(color_list)
            catid2color[catid] = color_list[color_index]
            catid2fontcolor[catid] = font_colormap(color_index)


    for ann in annotations:
        catid = int(ann[1])
        bbox = ann[2:]
        color = tuple(catid2color[catid])
        font_color = tuple(catid2fontcolor[catid])

        if len(bbox) == 4:
            # draw bbox
            cx, cy, w, h = bbox
            xmin = cx - 0.5*w
            ymin = cy - 0.5*h
            xmax = cx + 0.5*w
            ymax = cy + 0.5*h
            draw.line(
                [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin), (xmin, ymin)],
                width=width,
                fill=color,
            )
        else:
            logging.info("Error: The shape of bbox must be [M, 4]!")

        # draw label
        label = 'targets_' + str(catid)
        text = "{}".format(label)
        if tuple(map(int, PIL.__version__.split("."))) <= (10, 0, 0):
            tw, th = draw.textsize(text, font=font)
        else:
            left, top, right, bottom = draw.textbbox((0, 0), text, font)
            tw, th = right - left, bottom - top

        if ymin < th:
            draw.rectangle([(xmin, ymin), (xmin + tw + 4, ymin + th + 1)], fill=color)
            draw.text((xmin + 2, ymin - 2), text, fill=font_color, font=font)
        else:
            draw.rectangle([(xmin, ymin - th), (xmin + tw + 4, ymin + 1)], fill=color)
            draw.text((xmin + 2, ymin - th - 2), text, fill=font_color, font=font)

    return image