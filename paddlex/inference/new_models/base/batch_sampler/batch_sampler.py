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

from ..component import BaseComponent
from .batch_data import BatchData


class BaseBatchSampler(BaseComponent):

    def __init__(self, batch_size=1):
        self._batch_size = batch_size
        super().__init__()

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, bs):
        assert bs > 0
        self._batch_size = bs

    def __call__(self, *args, **kwargs):
        for batch, num in self.apply(*args, **kwargs):
            yield BatchData(
                {f"{self.name}.{k}": batch[0] for k in self.OUTPUT_KEYS}, num
            )

    @abstractmethod
    def apply(self, *args, **kwargs):
        raise NotImplementedError

    # def set_outputs(self, outputs):
    #     assert isinstance(outputs, dict)
    #     self.outputs = outputs
