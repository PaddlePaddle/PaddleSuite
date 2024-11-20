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

from ....utils import logging
from .component import BaseComponent


class BaseTransformer(BaseComponent):

    def __call__(self, batch_data):
        logging.debug(f"Call apply() func...")
        kwargs = {k: batch_data.get_by_key(v) for k, v in self.inputs}
        output = self.apply(**kwargs)
        if not output:
            return batch_data
        batch_data.update_by_key({f"{self.name}.{key}": output[key] for key in output})
        return batch_data

    @abstractmethod
    def apply(self, input):
        raise NotImplementedError
