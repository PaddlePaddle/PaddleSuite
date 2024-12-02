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

from ....utils.flags import INFER_BENCHMARK
from ...utils.benchmark import Timer


class BaseComponent(ABC):

    def __init__(self):
        self.name = getattr(self, "NAME", self.__class__.__name__)
        self.timer = Timer(self) if INFER_BENCHMARK else None

    @abstractmethod
    def __call__(self, batch_data):
        raise NotImplementedError

    @abstractmethod
    def apply(self, input):
        raise NotImplementedError
