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
import inspect
from abc import abstractmethod
import lazy_paddle as paddle
import numpy as np

from .....utils.flags import FLAGS_json_format_model
from .....utils import logging
from ....utils.pp_option import PaddlePredictorOption


class Copy2GPU:

    def __init__(self, input_handlers):
        super().__init__()
        self.input_handlers = input_handlers

    def __call__(self, x):
        for idx in range(len(x)):
            self.input_handlers[idx].reshape(x[idx].shape)
            self.input_handlers[idx].copy_from_cpu(x[idx])


class Copy2CPU:

    def __init__(self, output_handlers):
        super().__init__()
        self.output_handlers = output_handlers

    def __call__(self):
        output = []
        for out_tensor in self.output_handlers:
            batch = out_tensor.copy_to_cpu()
            output.append(batch)
        return output


class Infer:

    def __init__(self, predictor):
        super().__init__()
        self.predictor = predictor

    def __call__(self):
        self.predictor.run()


class BaseStaticInfer:
    """Predictor based on Paddle Inference"""

    def __init__(self, model_dir, model_prefix, option):
        super().__init__()
        self.model_dir = model_dir
        self.model_prefix = model_prefix
        self._update_option(option)

    def _update_option(self, option):
        if option:
            if self.option and option == self.option:
                return
            self._option = option
            self._reset()

    @property
    def option(self):
        return self._option if hasattr(self, "_option") else None

    @option.setter
    def option(self, option):
        self._update_option(option)

    def _reset(self):
        if not self.option:
            self.option = PaddlePredictorOption()
        logging.debug(f"Env: {self.option}")
        (
            predictor,
            input_handlers,
            output_handlers,
        ) = self._create()
        self.copy2gpu = Copy2GPU(input_handlers)
        self.copy2cpu = Copy2CPU(output_handlers)
        self.infer = Infer(predictor)
        self.option.changed = False

    def _create(self):
        """_create"""
        from lazy_paddle.inference import Config, create_predictor

        model_postfix = ".json" if FLAGS_json_format_model else ".pdmodel"
        model_file = (self.model_dir / f"{self.model_prefix}{model_postfix}").as_posix()
        params_file = (self.model_dir / f"{self.model_prefix}.pdiparams").as_posix()
        config = Config(model_file, params_file)

        config.enable_memory_optim()
        if self.option.device in ("gpu", "dcu"):
            if self.option.device == "gpu":
                config.exp_disable_mixed_precision_ops({"feed", "fetch"})
            config.enable_use_gpu(100, self.option.device_id)
            if self.option.device == "gpu":
                # NOTE: The pptrt settings are not aligned with those of FD.
                precision_map = {
                    "trt_int8": Config.Precision.Int8,
                    "trt_fp32": Config.Precision.Float32,
                    "trt_fp16": Config.Precision.Half,
                }
                if self.option.run_mode in precision_map.keys():
                    config.enable_tensorrt_engine(
                        workspace_size=(1 << 25) * self.option.batch_size,
                        max_batch_size=self.option.batch_size,
                        min_subgraph_size=self.option.min_subgraph_size,
                        precision_mode=precision_map[self.option.run_mode],
                        use_static=self.option.trt_use_static,
                        use_calib_mode=self.option.trt_calib_mode,
                    )

                    if self.option.shape_info_filename is not None:
                        if not os.path.exists(self.option.shape_info_filename):
                            config.collect_shape_range_info(
                                self.option.shape_info_filename
                            )
                            logging.info(
                                f"Dynamic shape info is collected into: {self.option.shape_info_filename}"
                            )
                        else:
                            logging.info(
                                f"A dynamic shape info file ( {self.option.shape_info_filename} ) already exists. \
        No need to generate again."
                            )
                        config.enable_tuned_tensorrt_dynamic_shape(
                            self.option.shape_info_filename, True
                        )
        elif self.option.device == "npu":
            config.enable_custom_device("npu")
        elif self.option.device == "xpu":
            pass
        elif self.option.device == "mlu":
            config.enable_custom_device("mlu")
        else:
            assert self.option.device == "cpu"
            config.disable_gpu()
            if "mkldnn" in self.option.run_mode:
                try:
                    config.enable_mkldnn()
                    if "bf16" in self.option.run_mode:
                        config.enable_mkldnn_bfloat16()
                except Exception as e:
                    logging.warning(
                        "MKL-DNN is not available. We will disable MKL-DNN."
                    )
                config.set_mkldnn_cache_capacity(-1)
            else:
                if hasattr(config, "disable_mkldnn"):
                    config.disable_mkldnn()

        # Disable paddle inference logging
        config.disable_glog_info()

        config.set_cpu_math_library_num_threads(self.option.cpu_threads)

        if self.option.device in ("cpu", "gpu"):
            if not (
                self.option.device == "gpu" and self.option.run_mode.startswith("trt")
            ):
                if hasattr(config, "enable_new_ir"):
                    config.enable_new_ir(self.option.enable_new_ir)
                if hasattr(config, "enable_new_executor"):
                    config.enable_new_executor()
                config.set_optimization_level(3)

        for del_p in self.option.delete_pass:
            config.delete_pass(del_p)

        if self.option.device in ("gpu", "dcu"):
            if paddle.is_compiled_with_rocm():
                # Delete unsupported passes in dcu
                config.delete_pass("conv2d_add_act_fuse_pass")
                config.delete_pass("conv2d_add_fuse_pass")

        predictor = create_predictor(config)

        # Get input and output handlers
        input_names = predictor.get_input_names()
        input_names.sort()
        input_handlers = []
        output_handlers = []
        for input_name in input_names:
            input_handler = predictor.get_input_handle(input_name)
            input_handlers.append(input_handler)
        output_names = predictor.get_output_names()
        for output_name in output_names:
            output_handler = predictor.get_output_handle(output_name)
            output_handlers.append(output_handler)
        return predictor, input_handlers, output_handlers

    def __call__(self, **kwargs):
        if self.option.changed:
            self._reset()
        batches = self.to_batch(**kwargs)
        self.copy2gpu(batches)
        self.infer()
        pred = self.copy2cpu()
        return self.format_output(pred)

    @property
    def benchmark(self):
        return {
            "Copy2GPU": self.copy2gpu,
            "Infer": self.infer,
            "Copy2CPU": self.copy2cpu,
        }

    @abstractmethod
    def to_batch(self):
        raise NotImplementedError

    @abstractmethod
    def format_output(self, pred):
        raise NotImplementedError