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

import inspect
from abc import ABC, abstractmethod
from types import GeneratorType

from ....utils.flags import INFER_BENCHMARK
from ....utils import logging
from ...utils.benchmark import Timer


class Param:
    def __init__(self, name, cmpt):
        self.name = name
        self.cmpt = cmpt

    def __repr__(self):
        return f"{self.cmpt.name}.{self.name}"


class OutParam(Param):
    pass


class InParam(Param):
    def fetch(self, param: OutParam):
        self.cmpt.set_dep(self, param)


class InOuts:
    def __getattr__(self, key):
        if key in self._keys:
            return self._keys.get(key)
        raise AttributeError(
            f"'{self._cmpt.name}.{self.__class__.__name__}' object has no attribute '{key}'"
        )

    def __repr__(self):
        _str = ""
        for key in self._keys:
            param = self._keys[key]
            _str += f"{param.cmpt.name}.{param.name}"
        return _str


class Outputs(InOuts):
    def __init__(self, cmpt):
        self._cmpt = cmpt
        self._keys = {}
        if cmpt.OUTPUT_KEYS:
            for key in cmpt.OUTPUT_KEYS:
                self._keys[key] = OutParam(key, cmpt)


class Inputs(InOuts):
    def __init__(self, cmpt):
        self._cmpt = cmpt
        self._keys = {}
        if cmpt.INPUT_KEYS:
            for key in cmpt.INPUT_KEYS:
                self._keys[key] = InParam(key, cmpt)

    def __iter__(self):
        for in_param, out_param in self._cmpt.dependencies:
            out_param_str = f"{out_param.cmpt.name}.{out_param.name}"
            yield f"{in_param.name}", f"{out_param_str}"

    # def __repr__(self):
    #     _str = ""
    #     for in_param, out_param in self._dependencies:
    #         out_param_str = f"{out_param.cmpt.name}.{out_param.name}"
    #         _str += f"{in_param.cmpt.name}.{in_param.name}: {out_param_str}\t"
    #     return _str


# class Dependencies:
#     def __init__(self):
#         pass

#     def add(self, )


class BaseComponent(ABC):

    INPUT_KEYS = None
    OUTPUT_KEYS = None

    def __init__(self):
        self.name = getattr(self, "NAME", self.__class__.__name__)
        self.inputs = Inputs(self)
        self.outputs = Outputs(self)
        self.dependencies = []

        if INFER_BENCHMARK:
            self.timer = Timer()
            self.apply = self.timer.watch_func(self.apply)

    def set_dep(self, in_param, out_param):
        self.dependencies.append((in_param, out_param))

    @classmethod
    def get_input_keys(cls) -> list:
        return cls.INPUT_KEYS

    @classmethod
    def get_output_keys(cls) -> list:
        return cls.OUTPUT_KEYS

    @abstractmethod
    def __call__(self, batch_data):
        raise NotImplementedError

    @abstractmethod
    def apply(self, input):
        raise NotImplementedError
