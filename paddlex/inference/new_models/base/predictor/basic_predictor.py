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

from typing import Union, Tuple, List, Dict, Any, Iterator
from abc import abstractmethod

from .....utils.subclass_register import AutoRegisterABCMetaClass
from .....utils.flags import (
    INFER_BENCHMARK,
    INFER_BENCHMARK_WARMUP,
)
from .....utils import logging
from ....utils.pp_option import PaddlePredictorOption
from ....utils.benchmark import benchmark
from ..batch_sampler import BaseBatchSampler
from .base_predictor import BasePredictor


class PredictionWrap:
    def __init__(self, data: Dict[str, List[Any]], num: int) -> None:
        assert isinstance(data, dict)
        for k in data:
            assert len(data[k]) == num, f"{len(data[k])} != {num} !"
        self._data = data
        self._keys = data.keys()

    def get_by_idx(self, idx: int) -> Dict[str, Any]:
        return {key: self._data[key][idx] for key in self._keys}


class BasicPredictor(
    BasePredictor,
    metaclass=AutoRegisterABCMetaClass,
):

    __is_base = True

    def __init__(
        self,
        model_dir: str,
        config: Dict[str, Any] = None,
        device: str = None,
        pp_option: PaddlePredictorOption = None,
    ) -> None:
        super().__init__(model_dir=model_dir, config=config)
        if not pp_option:
            pp_option = PaddlePredictorOption(model_name=self.model_name)
        if device:
            pp_option.device = device
        self.pp_option = pp_option
        self.batch_sampler = self._build_batch_sampler()
        self.result_class = self._get_result_class()
        logging.debug(f"{self.__class__.__name__}: {self.model_dir}")
        self.benchmark = benchmark

    def __call__(self, input: Any, **kwargs: Dict[str, Any]) -> Iterator[Any]:
        self.set_predictor(**kwargs)
        if self.benchmark:
            self.benchmark.start()
            if INFER_BENCHMARK_WARMUP > 0:
                output = self.apply(input)
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
            output = list(self.apply(input))
            self.benchmark.collect(len(output))
        else:
            yield from self.apply(input)

    def apply(self, input: Any) -> Iterator[Any]:
        """predict"""
        for batch_data in self.batch_sampler(input):
            prediction = self.process(batch_data)
            prediction = PredictionWrap(prediction, len(batch_data))
            for idx in range(len(batch_data)):
                yield self.result_class(prediction.get_by_idx(idx))

    def set_predictor(
        self,
        batch_size: int = None,
        device: str = None,
        pp_option: PaddlePredictorOption = None,
    ) -> None:
        if batch_size:
            self.batch_sampler.batch_size = batch_size
            self.pp_option.batch_size = batch_size
        if device and device != self.pp_option.device:
            self.pp_option.device = device
        if pp_option and pp_option != self.pp_option:
            self.pp_option = pp_option

    @abstractmethod
    def _build_batch_sampler(self) -> BaseBatchSampler:
        raise NotImplementedError

    @abstractmethod
    def process(self, batch_data: List[Any]) -> Dict[str, List[Any]]:
        raise NotImplementedError

    @abstractmethod
    def _get_result_class(self) -> type:
        raise NotImplementedError
