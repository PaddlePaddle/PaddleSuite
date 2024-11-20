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

from ...component import BaseComponent


class ResultPackager(BaseComponent):

    OUTPUT_KEYS = None

    def __init__(self, result_class):
        self._result_class = result_class
        self.INPUT_KEYS = result_class.INPUT_KEYS
        super().__init__()

    def __call__(self, batch_data):
        yield from self.apply(batch_data)

    def apply(self, batch_data):
        for idx in range(batch_data.num):
            single = batch_data.get_by_idx(idx)
            yield self._result_class({k: single[v] for k, v in self.inputs})

    def set_inputs(self, inputs):
        assert isinstance(inputs, dict)
        self.inputs = inputs
