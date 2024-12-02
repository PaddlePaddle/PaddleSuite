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

from abc import ABC, abstractmethod

from ....utils.flags import (
    INFER_BENCHMARK,
    INFER_BENCHMARK_ITER,
    INFER_BENCHMARK_DATA_SIZE,
)
from .component import BaseComponent


class BaseBatchSampler(BaseComponent):

    def __init__(self, batch_size=1):
        super().__init__()
        self._batch_size = batch_size
        self._benchmark = INFER_BENCHMARK
        self._benchmark_iter = INFER_BENCHMARK_ITER
        self._benchmark_data_size = INFER_BENCHMARK_DATA_SIZE

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, bs):
        assert bs > 0
        self._batch_size = bs

    def __call__(self, input):
        if input is None and self._benchmark:
            for _ in range(self._benchmark_iter):
                yield self._rand_batch(self._benchmark_data_size)
        else:
            yield from self.apply(input)

    @abstractmethod
    def apply(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _rand_batch(self, data_size):
        raise NotImplementedError
