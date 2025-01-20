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

import abc
from typing import Union, Sequence, Tuple, List, Any
import lazy_paddle as paddle
import numpy as np
from pathlib import Path

from ....utils.flags import FLAGS_json_format_model
from ....utils import logging
from ...utils.pp_option import PaddlePredictorOption
from ...utils.hpi import MBIConfig


CACHE_DIR = ".cache"


# XXX: Better use Paddle Inference API to do this
def _pd_dtype_to_np_dtype(pd_dtype):
    if pd_dtype == paddle.inference.DataType.FLOAT64:
        return np.float64
    elif pd_dtype == paddle.inference.DataType.FLOAT32:
        return np.float32
    elif pd_dtype == paddle.inference.DataType.INT64:
        return np.int64
    elif pd_dtype == paddle.inference.DataType.INT32:
        return np.int32
    elif pd_dtype == paddle.inference.DataType.UINT8:
        return np.uint8
    elif pd_dtype == paddle.inference.DataType.INT8:
        return np.int8
    else:
        raise TypeError(f"Unsupported data type: {pd_dtype}")


def _collect_trt_shape_range_info(
    model_file,
    model_params,
    gpu_id,
    shape_range_info_path,
    dynamic_shapes,
    dynamic_shape_input_data,
):
    dynamic_shape_input_data = dynamic_shape_input_data or {}

    config = paddle.inference.Config(model_file, model_params)
    config.enable_use_gpu(100, gpu_id)
    config.collect_shape_range_info(shape_range_info_path)
    # TODO: Add other needed options
    config.disable_glog_info()
    predictor = paddle.inference.create_predictor(config)

    input_names = predictor.get_input_names()
    for name in dynamic_shapes:
        if name not in input_names:
            raise ValueError(
                f"Invalid input name {repr(name)} found in `dynamic_shapes`"
            )
    for name in input_names:
        if name not in dynamic_shapes:
            raise ValueError(f"Input name {repr(name)} not found in `dynamic_shapes`")
    for name in dynamic_shape_input_data:
        if name not in input_names:
            raise ValueError(
                f"Invalid input name {repr(name)} found in `dynamic_shape_input_data`"
            )
    # It would be better to check if the shapes are valid.

    min_arrs, opt_arrs, max_arrs = {}, {}, {}
    for name, candidate_shapes in dynamic_shapes.items():
        # XXX: Currently we have no way to get the data type of the tensor
        # without creating an input handle.
        handle = predictor.get_input_handle(name)
        dtype = _pd_dtype_to_np_dtype(handle.type())
        min_shape, opt_shape, max_shape = candidate_shapes
        if name in dynamic_shape_input_data:
            min_arrs[name] = np.array(
                dynamic_shape_input_data[name][0], dtype=dtype
            ).reshape(min_shape)
            opt_arrs[name] = np.array(
                dynamic_shape_input_data[name][1], dtype=dtype
            ).reshape(opt_shape)
            max_arrs[name] = np.array(
                dynamic_shape_input_data[name][2], dtype=dtype
            ).reshape(max_shape)
        else:
            min_arrs[name] = np.ones(min_shape, dtype=dtype)
            opt_arrs[name] = np.ones(opt_shape, dtype=dtype)
            max_arrs[name] = np.ones(max_shape, dtype=dtype)

    # `opt_arrs` is used twice to ensure it is the most frequently used.
    for arrs in [min_arrs, opt_arrs, opt_arrs, max_arrs]:
        for name, arr in arrs.items():
            handle = predictor.get_input_handle(name)
            handle.reshape(arr.shape)
            handle.copy_from_cpu(arr)
        predictor.run()

    # HACK: The shape range info will be written to the file only when
    # `predictor` is garbage collected. It works in CPython, but it is
    # definitely a bad idea to count on the implementation-dependent behavior of
    # a garbage collector. Is there a more explicit and deterministic way to
    # handle this?


class PaddleCopy2GPU:
    def __init__(self, input_handlers):
        super().__init__()
        self.input_handlers = input_handlers

    def __call__(self, x):
        for idx in range(len(x)):
            self.input_handlers[idx].reshape(x[idx].shape)
            self.input_handlers[idx].copy_from_cpu(x[idx])


class PaddleCopy2CPU:
    def __init__(self, output_handlers):
        super().__init__()
        self.output_handlers = output_handlers

    def __call__(self):
        output = []
        for out_tensor in self.output_handlers:
            batch = out_tensor.copy_to_cpu()
            output.append(batch)
        return output


class PaddleModelInfer:
    def __init__(self, predictor):
        super().__init__()
        self.predictor = predictor

    def __call__(self):
        self.predictor.run()


class StaticInfer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, x: Sequence[np.ndarray]) -> List[np.ndarray]:
        raise NotImplementedError


class PaddleInfer(StaticInfer):
    def __init__(
        self,
        model_dir: str,
        model_prefix: str,
        option: PaddlePredictorOption,
    ) -> None:
        super().__init__()
        self.model_dir = model_dir
        self.model_prefix = model_prefix
        self._update_option(option)

    @property
    def option(self) -> PaddlePredictorOption:
        return self._option if hasattr(self, "_option") else None

    # TODO: We should re-evaluate whether allowing changes to `option` across
    # different calls provides any benefits.
    @option.setter
    def option(self, option: Union[None, PaddlePredictorOption]) -> None:
        if option:
            self._update_option(option)

    @property
    def benchmark(self):
        return {
            "Copy2GPU": self.copy2gpu,
            "Infer": self.infer,
            "Copy2CPU": self.copy2cpu,
        }

    def __call__(self, x: Sequence[np.ndarray]) -> List[np.ndarray]:
        self.copy2gpu(x)
        self.infer()
        pred = self.copy2cpu()
        return pred

    def _update_option(self, option: PaddlePredictorOption) -> None:
        if self.option and option == self.option:
            return
        self._option = option
        self._reset()

    def _reset(self) -> None:
        if not self.option:
            self.option = PaddlePredictorOption()
        logging.debug(f"Env: {self.option}")
        (
            predictor,
            input_handlers,
            output_handlers,
        ) = self._create()
        # TODO: Would a more generic name like `copy2device` be better here?
        self.copy2gpu = PaddleCopy2GPU(input_handlers)
        self.copy2cpu = PaddleCopy2CPU(output_handlers)
        self.infer = PaddleModelInfer(predictor)

    def _create(
        self,
    ) -> Tuple[
        "paddle.base.libpaddle.PaddleInferPredictor",
        "paddle.base.libpaddle.PaddleInferTensor",
        "paddle.base.libpaddle.PaddleInferTensor",
    ]:
        """_create"""
        from lazy_paddle.inference import Config, create_predictor

        if FLAGS_json_format_model:
            model_file = (self.model_dir / f"{self.model_prefix}.json").as_posix()
        # when FLAGS_json_format_model is not set, use inference.json if exist, otherwise inference.pdmodel
        else:
            model_file = self.model_dir / f"{self.model_prefix}.json"
            if model_file.exists():
                model_file = model_file.as_posix()
            # default by `pdmodel` suffix
            else:
                model_file = (
                    self.model_dir / f"{self.model_prefix}.pdmodel"
                ).as_posix()
        params_file = (self.model_dir / f"{self.model_prefix}.pdiparams").as_posix()
        config = Config(model_file, params_file)

        config.enable_memory_optim()
        if self.option.device_type in ("gpu", "dcu"):
            if self.option.device_type == "gpu":
                config.exp_disable_mixed_precision_ops({"feed", "fetch"})
            config.enable_use_gpu(100, self.option.device_id or 0)
            if self.option.device_type == "gpu":
                precision_map = {
                    "trt_int8": Config.Precision.Int8,
                    "trt_fp32": Config.Precision.Float32,
                    "trt_fp16": Config.Precision.Half,
                }
                if self.option.run_mode in precision_map.keys():
                    self._configure_trt(
                        config,
                        precision_map[self.option.run_mode],
                        model_file,
                        params_file,
                    )

        elif self.option.device_type == "npu":
            config.enable_custom_device("npu")
        elif self.option.device_type == "xpu":
            pass
        elif self.option.device_type == "mlu":
            config.enable_custom_device("mlu")
        else:
            assert self.option.device_type == "cpu"
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

        if self.option.disable_glog_info:
            config.disable_glog_info()

        config.set_cpu_math_library_num_threads(self.option.cpu_threads)

        if self.option.device_type in ("cpu", "gpu"):
            if not (
                self.option.device_type == "gpu"
                and self.option.run_mode.startswith("trt")
            ):
                if hasattr(config, "enable_new_ir"):
                    config.enable_new_ir(self.option.enable_new_ir)
                if hasattr(config, "enable_new_executor"):
                    config.enable_new_executor()
                config.set_optimization_level(3)

        for del_p in self.option.delete_pass:
            config.delete_pass(del_p)

        if self.option.device_type in ("gpu", "dcu"):
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

    def _configure_trt(self, config, precision_mode, model_file, params_file):
        config.set_optim_cache_dir(str(self.model_dir / CACHE_DIR))

        config.enable_tensorrt_engine(
            workspace_size=self.option.trt_max_workspace_size,
            max_batch_size=self.option.trt_max_batch_size,
            min_subgraph_size=self.option.trt_min_subgraph_size,
            precision_mode=precision_mode,
            use_static=self.option.trt_use_static,
            use_calib_mode=self.option.trt_use_calib_mode,
        )

        if self.option.trt_use_dynamic_shapes:
            if self.option.trt_collect_shape_range_info:
                # NOTE: We always use a shape range info file.
                if self.option.trt_shape_range_info_path is not None:
                    trt_shape_range_info_path = Path(
                        self.option.trt_shape_range_info_path
                    )
                else:
                    trt_shape_range_info_path = (
                        self.model_dir / CACHE_DIR / "shape_range_info.pbtxt"
                    )
                should_collect_shape_range_info = True
                if not trt_shape_range_info_path.exists():
                    trt_shape_range_info_path.parent.mkdir(parents=True, exist_ok=True)
                    logging.info(
                        f"Shape range info will be collected into {trt_shape_range_info_path}"
                    )
                elif self.option.trt_discard_cached_shape_range_info:
                    trt_shape_range_info_path.unlink()
                    logging.info(
                        f"The shape range info file ({trt_shape_range_info_path}) has been removed, and the shape range info will be re-collected."
                    )
                else:
                    logging.info(
                        f"A shape range info file ({trt_shape_range_info_path}) already exists. There is no need to collect the info again."
                    )
                    should_collect_shape_range_info = False
                if should_collect_shape_range_info:
                    _collect_trt_shape_range_info(
                        model_file,
                        params_file,
                        self.option.device_id or 0,
                        str(trt_shape_range_info_path),
                        self.option.trt_dynamic_shapes,
                        self.option.trt_dynamic_shape_input_data,
                    )
                config.enable_tuned_tensorrt_dynamic_shape(
                    str(trt_shape_range_info_path),
                    self.option.trt_allow_build_at_runtime,
                )
            else:
                if self.option.trt_dynamic_shapes is not None:
                    min_shapes, opt_shapes, max_shapes = {}, {}, {}
                    for (
                        key,
                        shapes,
                    ) in self.option.trt_dynamic_shapes.items():
                        min_shapes[key] = shapes[0]
                        opt_shapes[key] = shapes[1]
                        max_shapes[key] = shapes[2]
                        config.set_trt_dynamic_shape_info(
                            min_shapes, max_shapes, opt_shapes
                        )
                else:
                    raise RuntimeError("No dynamic shape information provided")


class MultibackendInfer(StaticInfer):
    def __init__(
        self,
        model_dir: str,
        model_prefix: str,
        config: MBIConfig,
    ) -> None:
        from .ultra_infer import RuntimeOption

        super().__init__()
        self._model_dir = model_dir
        self._model_prefix = model_prefix
        self._config = config
        self._ui_option = RuntimeOption()

    @property
    def model_dir(self) -> str:
        return self._model_dir

    @property
    def model_prefix(self) -> str:
        return self._model_prefix

    @property
    def config(self) -> MBIConfig:
        return self._config

    def __call__(self, x: Sequence[np.ndarray]) -> List[np.ndarray]:
        raise NotImplementedError

    def _update_ui_option(self, ui_option):
        pass

    def _build_ui_model(self, ui_option):
        pass
