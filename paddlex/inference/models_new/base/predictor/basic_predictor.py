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

from copy import deepcopy
from typing import Any, Dict, Iterator, Optional, Union

from pydantic import ValidationError

from .....utils import logging
from .....utils.device import get_default_device, parse_device
from .....utils.flags import INFER_BENCHMARK_WARMUP
from .....utils.subclass_register import AutoRegisterABCMetaClass
from ....utils.benchmark import benchmark
from ....utils.hpi import HPIInfo, MBIConfig
from ....utils.pp_option import PaddlePredictorOption
from ..common import MultibackendInfer, PaddleInfer
from .base_predictor import BasePredictor


class BasicPredictor(
    BasePredictor,
    metaclass=AutoRegisterABCMetaClass,
):
    """BasicPredictor."""

    __is_base = True

    def __init__(
        self,
        model_dir: str,
        config: Optional[Dict[str, Any]] = None,
        *,
        device: Optional[str] = None,
        use_paddle: bool = True,
        pp_option: Optional[PaddlePredictorOption] = None,
        mbi_config: Optional[Union[Dict[str, Any], MBIConfig]] = None,
    ) -> None:
        """Initializes the BasicPredictor.

        Args:
            model_dir (str): The directory where the model files are stored.
            config (Optional[Dict[str, Any]], optional): The model configuration
                dictionary. Defaults to None.
            device (Optional[str], optional): The device to run the inference
                engine on. Defaults to None.
            use_paddle (bool, optional): Whether to use Paddle Inference.
                Defaults to True.
            pp_option (Optional[PaddlePredictorOption], optional): The inference
                engine options. Defaults to None.
            mbi_config (Optional[Union[Dict[str, Any], MBIConfig]], optional):
                The multi-backend inference configuration dictionary.
                Defaults to None.
        """
        super().__init__(model_dir=model_dir, config=config)

        self._use_paddle = use_paddle
        if use_paddle:
            self._pp_option = self._prepare_pp_option(pp_option, device)
        else:
            if mbi_config is None:
                raise ValueError(
                    "`mbi_config` must not be None when not using Paddle Inference."
                )
            self._mbi_config = self._prepare_mbi_config(mbi_config, device)

        logging.debug(f"{self.__class__.__name__}: {self.model_dir}")
        self.benchmark = benchmark

    @property
    def pp_option(self) -> PaddlePredictorOption:
        if not hasattr(self, "_pp_option"):
            raise AttributeError(f"{repr(self)} has no attribute 'pp_option'.")
        return self._pp_option

    @property
    def mbi_config(self) -> MBIConfig:
        if not hasattr(self, "_mbi_config"):
            raise AttributeError(f"{repr(self)} has no attribute 'mbi_config'.")
        return self._mbi_config

    @property
    def use_paddle(self) -> bool:
        return self.use_paddle

    def __call__(
        self,
        input: Any,
        batch_size: Optional[int] = None,
        **kwargs: Any,
    ) -> Iterator[Any]:
        """
        Predict with the input data.

        Args:
            input (Any): The input data to be predicted.
            batch_size (int, optional): The batch size to use. Defaults to None.
            **kwargs (Dict[str, Any]): Additional keyword arguments to set up predictor.

        Returns:
            Iterator[Any]: An iterator yielding the prediction output.
        """
        self.set_predictor(batch_size)
        if self.benchmark:
            self.benchmark.start()
            if INFER_BENCHMARK_WARMUP > 0:
                output = self.apply(input, **kwargs)
                warmup_num = 0
                for _ in range(INFER_BENCHMARK_WARMUP):
                    try:
                        next(output)
                        warmup_num += 1
                    except StopIteration:
                        logging.warning(
                            f"There are only {warmup_num} batches in input data, but `INFER_BENCHMARK_WARMUP` has been set to {INFER_BENCHMARK_WARMUP}."
                        )
                        break
                self.benchmark.warmup_stop(warmup_num)
            output = list(self.apply(input, **kwargs))
            self.benchmark.collect(len(output))
        else:
            yield from self.apply(input, **kwargs)

    def set_predictor(
        self,
        batch_size: Optional[int] = None,
    ) -> None:
        """
        Sets the predictor configuration.

        Args:
            batch_size (Optional[int], optional): The batch size to use. Defaults to None.

        Returns:
            None
        """
        if batch_size:
            self.batch_sampler.batch_size = batch_size

    def get_hpi_info(self):
        if "Hpi" not in self.config:
            return None
        try:
            return HPIInfo.model_validate(self.config["Hpi"])
        except ValidationError as e:
            logging.exception("The HPI info in the model config file is invalid.")
            raise RuntimeError(f"Invalid HPI info: {str(e)}") from e

    def create_static_infer(self):
        if self.use_paddle:
            return PaddleInfer(self.model_dir, self.MODEL_FILE_PREFIX, self._pp_option)
        else:
            return MultibackendInfer(
                self.model_dir, self.MODEL_FILE_PREFIX, self._mbi_config
            )

    def _prepare_pp_option(
        self,
        pp_option: Optional[PaddlePredictorOption],
        device: Optional[str],
    ) -> PaddlePredictorOption:
        if pp_option is None or device is not None:
            device_info = self._get_device_info(device)
        else:
            device_info = None
        if pp_option is None:
            pp_option = PaddlePredictorOption(model_name=self.model_name)
        else:
            # To avoid mutating the original input
            pp_option = deepcopy(pp_option)
        if device_info:
            pp_option.device_type = device_info[0]
            pp_option.device_id = device_info[1]
        hpi_info = self.get_hpi_info()
        if hpi_info is not None:
            logging.debug("HPI info: %s", hpi_info)
            hpi_info = hpi_info.model_dump(exclude_unset=True)
            trt_dynamic_shapes = (
                hpi_info.get("backend_configs", {})
                .get("paddle_infer", {})
                .get("trt_dynamic_shapes", None)
            )
            if trt_dynamic_shapes:
                pp_option.trt_dynamic_shapes = trt_dynamic_shapes
            trt_dynamic_shape_input_data = (
                hpi_info.get("backend_configs", {})
                .get("paddle_infer", {})
                .get("trt_dynamic_shape_input_data", None)
            )
            if trt_dynamic_shape_input_data:
                pp_option.trt_dynamic_shape_input_data = trt_dynamic_shape_input_data
        return pp_option

    def _prepare_mbi_config(
        self,
        mbi_config: Union[Dict[str, Any], MBIConfig],
        device: Optional[str],
    ) -> MBIConfig:
        # TODO: Avoid extra copying
        if not isinstance(mbi_config, MBIConfig):
            mbi_config = MBIConfig.model_validate(mbi_config)
        if device is not None:
            device_type, device_id = self._get_device_info(device)
            return mbi_config.model_copy(
                update={"device_type": device_type, "device_id": device_id}
            )
        else:
            return mbi_config.model_copy()

    # Should this be static?
    def _get_device_info(self, device):
        if device is None:
            device = get_default_device()
        device_type, device_ids = parse_device(device)
        if device_ids is not None:
            device_id = device_ids[0]
        else:
            device_id = None
        if device_ids and len(device_ids) > 1:
            logging.debug("Got multiple device IDs. Using the first one: %d", device_id)
        return device_type, device_id
