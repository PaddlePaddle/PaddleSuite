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

import yaml
import codecs
from pathlib import Path
from abc import abstractmethod

from ...utils.subclass_register import AutoRegisterABCMetaClass
from ..components.base import BaseComponent, ComponentsEngine
from ..utils.process_hook import generatorable_method


class BasePredictor(BaseComponent, metaclass=AutoRegisterABCMetaClass):
    __is_base = True

    INPUT_KEYS = "x"
    OUTPUT_KEYS = None

    KEEP_INPUT = False

    MODEL_FILE_PREFIX = "inference"

    def __init__(self, model_dir, config=None, device="gpu", **kwargs):
        super().__init__()
        self.model_dir = Path(model_dir)
        self.config = config if config else self.load_config(self.model_dir)
        self.device = device
        self.kwargs = kwargs
        self.components = self._build_components()
        self.engine = ComponentsEngine(self.components)
        # alias predict() to the __call__()
        self.predict = self.__call__

    @classmethod
    def load_config(cls, model_dir):
        config_path = model_dir / f"{cls.MODEL_FILE_PREFIX}.yml"
        with codecs.open(config_path, "r", "utf-8") as file:
            dic = yaml.load(file, Loader=yaml.FullLoader)
        return dic

    def apply(self, x):
        """predict"""
        yield from self._generate_res(self.engine(x))

    @generatorable_method
    def _generate_res(self, data):
        return self._pack_res(data)

    @abstractmethod
    def _build_components(self):
        raise NotImplementedError

    @abstractmethod
    def _pack_res(self, data):
        raise NotImplementedError
