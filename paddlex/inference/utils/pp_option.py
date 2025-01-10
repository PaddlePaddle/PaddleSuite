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
from typing import Dict, List

from ...utils.device import parse_device, set_env_for_device, get_default_device
from ...utils import logging
from .new_ir_blacklist import NEWIR_BLOCKLIST


class PaddlePredictorOption(object):
    """Paddle Inference Engine Option"""

    # NOTE: TRT modes start with `trt_`
    SUPPORT_RUN_MODE = (
        "paddle",
        "trt_fp32",
        "trt_fp16",
        "trt_int8",
        "mkldnn",
        "mkldnn_bf16",
    )
    SUPPORT_DEVICE = ("gpu", "cpu", "npu", "xpu", "mlu", "dcu", "gcu")

    def __init__(self, model_name=None, **kwargs):
        super().__init__()
        self.model_name = model_name
        self._cfg = {}
        self._init_option(**kwargs)
        self._changed = False

    @property
    def changed(self):
        return self._changed

    @changed.setter
    def changed(self, v):
        assert isinstance(v, bool)
        self._changed = v

    def _init_option(self, **kwargs):
        for k, v in kwargs.items():
            if self._has_setter(k):
                setattr(self, k, v)
            else:
                raise Exception(
                    f"{k} is not supported to set! The supported option is: {self._get_settable_attributes()}"
                )
        for k, v in self._get_default_config().items():
            self._cfg.setdefault(k, v)

    def _get_default_config(self):
        """get default config"""
        device_type, device_id = parse_device(get_default_device())
        return {
            "run_mode": "paddle",
            "device": device_type,
            "device_id": 0 if device_id is None else device_id[0],
            "cpu_threads": 8,
            "delete_pass": [],
            "enable_new_ir": True if self.model_name not in NEWIR_BLOCKLIST else False,
            "disable_glog_info": True,
            "trt_max_workspace_size": 1 << 30,  # only for trt
            "trt_max_batch_size": 32,  # only for trt
            "trt_min_subgraph_size": 3,  # only for trt
            "trt_use_static": False,  # only for trt
            "trt_use_calib_mode": False,  # only for trt
            "trt_use_dynamic_shapes": True,  # only for trt
            "trt_collect_shapes": False,  # only for trt
            "trt_dynamic_shapes": None,  # only for trt
            "trt_dynamic_shape_input_data": None,  # only for trt
            "trt_shape_range_info_path": None,  # only for trt
            "trt_allow_build_at_runtime": True,  # only for trt
        }

    def _update(self, k, v):
        self._cfg[k] = v
        self.changed = True

    @property
    def run_mode(self):
        return self._cfg["run_mode"]

    @run_mode.setter
    def run_mode(self, run_mode: str):
        """set run mode"""
        if run_mode not in self.SUPPORT_RUN_MODE:
            support_run_mode_str = ", ".join(self.SUPPORT_RUN_MODE)
            raise ValueError(
                f"`run_mode` must be {support_run_mode_str}, but received {repr(run_mode)}."
            )
        self._update("run_mode", run_mode)

    @property
    def device_type(self):
        return self._cfg["device"]

    @property
    def device_id(self):
        return self._cfg["device_id"]

    @property
    def device(self):
        return self._cfg["device"]

    @device.setter
    def device(self, device: str):
        """set device"""
        if not device:
            return
        device_type, device_ids = parse_device(device)
        if device_type not in self.SUPPORT_DEVICE:
            support_run_mode_str = ", ".join(self.SUPPORT_DEVICE)
            raise ValueError(
                f"The device type must be one of {support_run_mode_str}, but received {repr(device_type)}."
            )
        self._update("device", device_type)
        device_id = device_ids[0] if device_ids is not None else 0
        self._update("device_id", device_id)
        set_env_for_device(device)
        if device_type not in ("cpu"):
            if device_ids is None or len(device_ids) > 1:
                logging.debug(f"The device ID has been set to {device_id}.")
        # XXX(gaotingquan): set flag to accelerate inference in paddle 3.0b2
        if device_type in ("gpu", "cpu"):
            os.environ["FLAGS_enable_pir_api"] = "1"

    @property
    def cpu_threads(self):
        return self._cfg["cpu_threads"]

    @cpu_threads.setter
    def cpu_threads(self, cpu_threads):
        """set cpu threads"""
        if not isinstance(cpu_threads, int) or cpu_threads < 1:
            raise Exception()
        self._update("cpu_threads", cpu_threads)

    @property
    def delete_pass(self):
        return self._cfg["delete_pass"]

    @delete_pass.setter
    def delete_pass(self, delete_pass):
        self._update("delete_pass", delete_pass)

    @property
    def enable_new_ir(self):
        return self._cfg["enable_new_ir"]

    @enable_new_ir.setter
    def enable_new_ir(self, enable_new_ir: bool):
        """set run mode"""
        self._update("enable_new_ir", enable_new_ir)

    @property
    def trt_max_workspace_size(self):
        return self._cfg["trt_max_workspace_size"]

    @trt_max_workspace_size.setter
    def trt_max_workspace_size(self, trt_max_workspace_size):
        self._update("trt_max_workspace_size", trt_max_workspace_size)

    @property
    def trt_max_batch_size(self):
        return self._cfg["trt_max_batch_size"]

    @trt_max_batch_size.setter
    def trt_max_batch_size(self, trt_max_batch_size):
        self._update("trt_max_batch_size", trt_max_batch_size)

    @property
    def trt_min_subgraph_size(self):
        return self._cfg["trt_min_subgraph_size"]

    @trt_min_subgraph_size.setter
    def trt_min_subgraph_size(self, trt_min_subgraph_size: int):
        """set min subgraph size"""
        if not isinstance(trt_min_subgraph_size, int):
            raise Exception()
        self._update("trt_min_subgraph_size", trt_min_subgraph_size)

    @property
    def trt_use_static(self):
        return self._cfg["trt_use_static"]

    @trt_use_static.setter
    def trt_use_static(self, trt_use_static):
        """set trt use static"""
        self._update("trt_use_static", trt_use_static)

    @property
    def trt_use_calib_mode(self):
        return self._cfg["trt_use_calib_mode"]

    @trt_use_calib_mode.setter
    def trt_use_calib_mode(self, trt_use_calib_mode):
        """set trt calib mode"""
        self._update("trt_use_calib_mode", trt_use_calib_mode)

    @property
    def trt_use_dynamic_shapes(self):
        return self._cfg["trt_use_dynamic_shapes"]

    @trt_use_dynamic_shapes.setter
    def trt_use_dynamic_shapes(self, trt_use_dynamic_shapes):
        self._update("trt_use_dynamic_shapes", trt_use_dynamic_shapes)

    @property
    def trt_collect_shapes(self):
        return self._cfg["trt_collect_shapes"]

    @trt_collect_shapes.setter
    def trt_collect_shapes(self, trt_collect_shapes):
        self._update("trt_collect_shapes", trt_collect_shapes)

    @property
    def trt_dynamic_shapes(self):
        return self._cfg["trt_dynamic_shapes"]

    @trt_dynamic_shapes.setter
    def trt_dynamic_shapes(self, trt_dynamic_shapes: Dict[str, List[List[int]]]):
        assert isinstance(trt_dynamic_shapes, dict)
        for input_k in trt_dynamic_shapes:
            assert isinstance(trt_dynamic_shapes[input_k], list)
        self._update("trt_dynamic_shapes", trt_dynamic_shapes)

    @property
    def trt_dynamic_shape_input_data(self):
        return self._cfg["trt_dynamic_shape_input_data"]

    @trt_dynamic_shape_input_data.setter
    def trt_dynamic_shape_input_data(
        self, trt_dynamic_shape_input_data: Dict[str, List[float]]
    ):
        self._update("trt_dynamic_shape_input_data", trt_dynamic_shape_input_data)

    @property
    def trt_shape_range_info_path(self):
        return self._cfg["trt_shape_range_info_path"]

    @trt_shape_range_info_path.setter
    def trt_shape_range_info_path(self, trt_shape_range_info_path: str):
        """set shape info filename"""
        self._update("trt_shape_range_info_path", trt_shape_range_info_path)

    @property
    def trt_allow_build_at_runtime(self):
        return self._cfg["trt_allow_build_at_runtime"]

    @trt_allow_build_at_runtime.setter
    def trt_allow_build_at_runtime(self, trt_allow_build_at_runtime):
        self._update("trt_allow_build_at_runtime", trt_allow_build_at_runtime)

    def get_support_run_mode(self):
        """get supported run mode"""
        return self.SUPPORT_RUN_MODE

    def get_support_device(self):
        """get supported device"""
        return self.SUPPORT_DEVICE

    def __str__(self):
        return ",  ".join([f"{k}: {v}" for k, v in self._cfg.items()])

    def __getattr__(self, key):
        if key not in self._cfg:
            raise Exception(f"The key ({key}) is not found in cfg: \n {self._cfg}")
        return self._cfg.get(key)

    def __eq__(self, obj):
        if isinstance(obj, PaddlePredictorOption):
            return obj._cfg == self._cfg
        return False

    def _has_setter(self, attr):
        prop = getattr(self.__class__, attr, None)
        return isinstance(prop, property) and prop.fset is not None

    def _get_settable_attributes(self):
        return [
            name
            for name, prop in vars(self.__class__).items()
            if isinstance(prop, property) and prop.fset is not None
        ]
