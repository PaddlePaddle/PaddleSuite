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

import functools


def batchable(func):
    @functools.wraps(func)
    def wrap(self, **batch_kwargs):
        outputs = {}
        keys = list(batch_kwargs.keys())
        single_kwargs = [
            dict(zip(keys, values)) for values in zip(*batch_kwargs.values())
        ]
        for kwargs in single_kwargs:
            single_output = func(self, **kwargs)
            for k, v in single_output.items():
                if k not in outputs:
                    outputs[k] = []
                outputs[k].append(v)
        return outputs

    return wrap
