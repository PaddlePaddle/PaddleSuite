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


class BatchData(object):

    def __init__(self, data, num):
        self._data = data
        self._num = num

    @property
    def num(self):
        return self._num

    def get_by_key(self, key):
        assert key in self._data, f"{key}, {list(self._data.keys())}"
        return self._data[key]

    def get_by_idx(self, idx):
        assert idx <= self.num
        return {k: v[idx] for k, v in self._data.items()}

    def update_by_key(self, output):
        assert isinstance(output, dict)
        for k, v in output.items():
            assert isinstance(v, list) and len(v) == self.num
            self._data[k] = v
