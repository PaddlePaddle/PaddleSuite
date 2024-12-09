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
from abc import ABC, abstractmethod

from ....utils.flags import (
    INFER_BENCHMARK,
    INFER_BENCHMARK_ITER,
    INFER_BENCHMARK_DATA_SIZE,
)


class BaseBatchSampler:

    def __init__(self, batch_size: int = 1) -> None:
        super().__init__()
        self._batch_size = batch_size
        self._benchmark = INFER_BENCHMARK
        self._benchmark_iter = INFER_BENCHMARK_ITER
        self._benchmark_data_size = INFER_BENCHMARK_DATA_SIZE

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @batch_size.setter
    def batch_size(self, bs: int) -> None:
        assert bs > 0
        self._batch_size = bs

    def __call__(self, input: Any) -> Iterator[List[Any]]:
        if input is None and self._benchmark:
            for _ in range(self._benchmark_iter):
                yield self._rand_batch(self._benchmark_data_size)
        else:
            yield from self.apply(input)

    @abstractmethod
    def apply(self, *args: Tuple[Any], **kwargs: Dict[str, Any]) -> Iterator[list]:
        raise NotImplementedError

    @abstractmethod
    def _rand_batch(self, data_size: int) -> List[Any]:
        raise NotImplementedError
