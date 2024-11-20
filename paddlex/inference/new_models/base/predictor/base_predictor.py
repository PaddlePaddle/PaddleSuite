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

from pathlib import Path
from abc import abstractmethod, ABC

from ....utils.io import YAMLReader


class BasePredictor(ABC):

    MODEL_FILE_PREFIX = "inference"

    def __init__(self, model_dir, config=None):
        super().__init__()
        self.model_dir = Path(model_dir)
        self.config = config if config else self.load_config(self.model_dir)

        # alias predict() to the __call__()
        self.predict = self.__call__
        self.pkg_res = True
        self.benchmark = None

    @property
    def config_path(self):
        return self.get_config_path(self.model_dir)

    @property
    def model_name(self) -> str:
        return self.config["Global"]["model_name"]

    @classmethod
    def get_config_path(cls, model_dir):
        return model_dir / f"{cls.MODEL_FILE_PREFIX}.yml"

    @classmethod
    def load_config(cls, model_dir):
        yaml_reader = YAMLReader()
        return yaml_reader.read(cls.get_config_path(model_dir))

    @property
    def package_result(self):
        return self._pkg_res

    @package_result.setter
    def package_result(self, pkg_res):
        assert isinstance(pkg_res, bool)
        self._pkg_res = pkg_res

    @abstractmethod
    def __call__(self, input, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def apply(self, input):
        raise NotImplementedError

    @abstractmethod
    def set_predictor(self):
        raise NotImplementedError
