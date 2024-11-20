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

from abc import abstractmethod

from ......utils.subclass_register import AutoRegisterABCMetaClass
from ......utils.flags import (
    INFER_BENCHMARK,
    INFER_BENCHMARK_WARMUP,
)
from ......utils import logging
from .....utils.pp_option import PaddlePredictorOption
from .....utils.benchmark import Benchmark
from ..base_predictor import BasePredictor
from .transformer_engine import TransformerEngine
from .result_packager import ResultPackager


class BasicPredictor(
    BasePredictor,
    metaclass=AutoRegisterABCMetaClass,
):

    __is_base = True

    def __init__(self, model_dir, config=None, device=None, pp_option=None):
        super().__init__(model_dir=model_dir, config=config)
        if not pp_option:
            pp_option = PaddlePredictorOption(model_name=self.model_name)
        if device:
            pp_option.device = device
        self.pp_option = pp_option

        self.batch_sampler = self._build_batch_sampler()
        self.result_packager = self._build_result_packager()
        self.transformers = {}
        self._build_transformers()
        self._set_dataflow()
        self.engine = TransformerEngine(self.transformers)
        self.rtn_res = True
        logging.debug(f"{self.__class__.__name__}: {self.model_dir}")

        if INFER_BENCHMARK:
            self.benchmark = Benchmark(self.transformers)

    def __call__(self, input, **kwargs):
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

    def apply(self, input):
        """predict"""
        for batch in self.batch_sampler(input):
            batch_data = self.engine(batch)
            if self.rtn_res:
                yield from self.result_packager(batch_data)
            else:
                yield batch_data

    def set_predictor(self, batch_size=None, device=None, pp_option=None):
        if batch_size:
            self.batch_sampler.batch_size = batch_size
            self.pp_option.batch_size = batch_size
        if device and device != self.pp_option.device:
            self.pp_option.device = device
        if pp_option and pp_option != self.pp_option:
            self.pp_option = pp_option

    def _add_transformer(self, cmp):
        self.transformers[cmp.name] = cmp

    def __getattr__(self, cmp):
        if cmp in self.transformers:
            return self.transformers.get(cmp)
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{cmp}'"
        )

    def _build_result_packager(self):
        return ResultPackager(self._get_result_class())

    @abstractmethod
    def _build_batch_sampler(self):
        raise NotImplementedError

    @abstractmethod
    def _get_result_class(self):
        raise NotImplementedError

    @abstractmethod
    def _build_transformers(self):
        raise NotImplementedError

    @abstractmethod
    def _set_dataflow(self):
        raise NotImplementedError
