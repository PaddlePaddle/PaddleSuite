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

import json
import os
import os.path as osp
from collections import defaultdict
from pathlib import Path
from nuscenes import NuScenes
import pickle

from ..base import BaseDatasetChecker
from ...utils.errors import DatasetFileNotFoundError
from ...utils.misc import abspath
from .model_list import MODELS


class BEVFusionDatasetChecker(BaseDatasetChecker):
    entities = MODELS

    def check_dataset(self, dataset_dir):
        dataset_dir = abspath(dataset_dir)
        max_sample_num = 5

        if not osp.exists(dataset_dir) or not osp.isdir(dataset_dir):
            raise DatasetFileNotFoundError(file_path=dataset_dir)

        anno_file = osp.join(dataset_dir, "nuscenes_infos_train.pkl")
        if not osp.exists(anno_file):
            raise DatasetFileNotFoundError(file_path=anno_file)
        train_mate = self.get_data(anno_file, max_sample_num)

        anno_file = osp.join(dataset_dir, "nuscenes_infos_val.pkl")
        if not osp.exists(anno_file):
            raise DatasetFileNotFoundError(file_path=anno_file)
        val_mate = self.get_data(anno_file, max_sample_num)

        meta = {"train_mate": train_mate, "val_mate": val_mate}
        return meta

    def get_data(self, ann_file, max_sample_num):
        infos = self.data_infos(ann_file, max_sample_num)
        meta = []
        for info in infos:
            image_paths = []
            cam_orders = [
                "CAM_FRONT_LEFT",
                "CAM_FRONT",
                "CAM_FRONT_RIGHT",
                "CAM_BACK_RIGHT",
                "CAM_BACK",
                "CAM_BACK_LEFT",
            ]
            for cam_type in cam_orders:
                cam_info = info["cams"][cam_type]
                cam_data_path = cam_info["data_path"]
                image_paths.append(cam_data_path)

            meta.append(
                {
                    "sample_idx": info["token"],
                    "lidar_path": info["lidar_path"],
                    "image_paths": image_paths,
                }
            )
        return meta

    def data_infos(self, ann_file, max_sample_num):
        data = pickle.load(open(ann_file, "rb"))
        data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))
        data_infos = data_infos[:max_sample_num]
        return data_infos

    def get_show_type(self) -> str:
        """get the show type of dataset

        Returns:
            str: show type
        """
        return "path for images and lidar"

    def get_dataset_type(self) -> str:
        """return the dataset type

        Returns:
            str: dataset type
        """
        return "NuscenesMMDataset"
