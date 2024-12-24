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
from os import PathLike
from pathlib import Path
from typing import (
    Any,
    Dict,
    Final,
    Iterator,
    Optional,
    TypedDict,
    Union,
)

import ultra_infer as ui
from ultra_infer.model import BaseUltraInferModel
from paddlex.inference.common.reader import ReadImage
from paddlex.inference.models_new import BasePredictor
from paddlex.inference.utils.new_ir_blacklist import NEWIR_BLOCKLIST
from paddlex.utils import device as device_helper
from paddlex.utils import logging
from paddlex.utils.subclass_register import AutoRegisterABCMetaClass
from typing_extensions import assert_never

from paddlex_hpi._config import HPIConfig
from paddlex_hpi._utils.typing import Backend

HPI_CONFIG_KEY: Final[str] = "Hpi"


class HPIParams(TypedDict, total=False):
    serial_number: Optional[str]
    update_license: bool
    config: Dict[str, Any]


class HPPredictor(BasePredictor, metaclass=AutoRegisterABCMetaClass):
    __is_base = True

    def __init__(
        self,
        model_dir: Union[str, PathLike],
        config: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
        hpi_params: Optional[HPIParams] = None,
    ) -> None:
        super().__init__(model_dir=model_dir, config=config)
        self._device = device or device_helper.get_default_device()
        self._hpi_params = hpi_params or {}
        self._hpi_config = self._get_hpi_config()
        self._ui_model = self.build_ui_model()
        self._data_reader = self._build_data_reader()

    def __call__(self, input: Any, **kwargs: dict[str, Any]) -> Iterator[Any]:
        self.set_predictor(**kwargs)
        yield from self.apply(input)

    @property
    def model_path(self) -> Path:
        return self.model_dir / f"{self.MODEL_FILE_PREFIX}.pdmodel"

    @property
    def params_path(self) -> Path:
        return self.model_dir / f"{self.MODEL_FILE_PREFIX}.pdiparams"

    def set_predictor(self, **kwargs: Any) -> None:
        if "device" in kwargs:
            device = kwargs.pop("device")
            if device is not None:
                if device != self._device:
                    raise RuntimeError("Currently, changing devices is not supported.")
        if "batch_size" in kwargs:
            self.batch_sampler.batch_size = kwargs.pop("batch_size")
        if kwargs:
            raise TypeError(f"Unexpected arguments: {kwargs}")

    def build_ui_model(self) -> BaseUltraInferModel:
        option = self._create_ui_option()
        return self._build_ui_model(option)

    def _get_hpi_config(self) -> HPIConfig:
        if HPI_CONFIG_KEY not in self.config:
            logging.debug("Key %r not found in the config", HPI_CONFIG_KEY)
        hpi_config = HPIConfig.model_validate(
            {
                **self.config.get(HPI_CONFIG_KEY, {}),
                **self._hpi_params.get("config", {}),
            }
        )
        return hpi_config

    def _get_selected_backend(self) -> Backend:
        device_type, _ = device_helper.parse_device(self._device)
        backend = self._hpi_config.get_selected_backend(self.model_name, device_type)
        return backend

    def _create_ui_option(self) -> ui.RuntimeOption:
        option = ui.RuntimeOption()
        # HACK: Disable new IR for models that are known to have issues with the
        # new IR.
        if self.model_name in NEWIR_BLOCKLIST:
            option.paddle_infer_option.enable_new_ir = False
        device_type, device_ids = device_helper.parse_device(self._device)
        if device_type == "cpu":
            pass
        elif device_type == "gpu":
            if device_ids is None:
                device_ids = [0]
            if len(device_ids) > 1:
                logging.warning(
                    "Multiple devices are specified (%s), but only the first one will be used.",
                    self._device,
                )
            option.use_gpu(device_ids[0])
        else:
            assert_never(device_type)
        backend, backend_config = self._hpi_config.get_backend_and_config(
            model_name=self.model_name, device_type=device_type
        )
        logging.info("Backend: %s", backend)
        logging.info("Backend config: %s", backend_config)
        backend_config.update_ui_option(option, self.model_dir)
        return option

    @abc.abstractmethod
    def _build_ui_model(self, option: ui.RuntimeOption) -> BaseUltraInferModel:
        raise NotImplementedError

    @abc.abstractmethod
    def _build_data_reader(self):
        raise NotImplementedError


class CVPredictor(HPPredictor):
    def _build_data_reader(self):
        return ReadImage(format="BGR")


class TSPredictor(HPPredictor):
    def _build_data_reader(self):
        return None
