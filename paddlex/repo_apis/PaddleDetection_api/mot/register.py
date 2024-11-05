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
from pathlib import Path

from ...base.register import register_model_info, register_suite_info
from .model import MOTModel
from .config import MOTConfig
from .runner import MOTRunner

REPO_ROOT_PATH = os.environ.get("PADDLE_PDX_PADDLEDETECTION_PATH")
PDX_CONFIG_DIR = osp.abspath(osp.join(osp.dirname(__file__), "..", "configs"))
HPI_CONFIG_DIR = Path(__file__).parent.parent.parent.parent / "utils" / "hpi_configs"

register_suite_info(
    {
        "suite_name": "MOT",
        "model": MOTModel,
        "runner": MOTRunner,
        "config": MOTConfig,
        "runner_root_path": REPO_ROOT_PATH,
    }
)

################ Models Using Universal Config ################

register_model_info(
    {
        "model_name": "ByteTrack_PP-YOLOE_L",
        "suite": "MOT",
        "config_path": osp.join(PDX_CONFIG_DIR, "ByteTrack_PP-YOLOE_L.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export", "infer"],
        "supported_dataset_types": ["COCODetDataset"],
        "supported_train_opts": {
            "device": ["cpu", "gpu_nxcx", "xpu", "npu", "mlu"],
            "dy2st": False,
            "amp": ["OFF"],
        },
        "hpi_config_path": HPI_CONFIG_DIR / "ByteTrack_PP-YOLOE_L.yaml",
    }
)

register_model_info(
    {
        "model_name": "DeepSORT_PP-YOLOE_ResNet",
        "suite": "MOT",
        "config_path": osp.join(PDX_CONFIG_DIR, "DeepSORT_PP-YOLOE_ResNet.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export", "infer"],
        "supported_dataset_types": ["COCODetDataset"],
        "supported_train_opts": {
            "device": ["cpu", "gpu_nxcx", "xpu", "npu", "mlu"],
            "dy2st": False,
            "amp": ["OFF"],
        },
        "hpi_config_path": HPI_CONFIG_DIR / "DeepSORT_PP-YOLOE_ResNet.yaml",
    }
)

register_model_info(
    {
        "model_name": "FairMOT-DLA-34",
        "suite": "MOT",
        "config_path": osp.join(PDX_CONFIG_DIR, "FairMOT-DLA-34.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export", "infer"],
        "supported_dataset_types": ["MOTDataSet"],
        "supported_train_opts": {
            "device": ["cpu", "gpu_nxcx", "xpu", "npu", "mlu"],
            "dy2st": False,
            "amp": ["OFF"],
        },
        "hpi_config_path": HPI_CONFIG_DIR / "FairMOT-DLA-34.yaml",
    }
)