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
from pathlib import Path
from abc import abstractmethod, ABC

from ....utils.io import YAMLReader


class BasePredictor(ABC):
    """BasePredictor."""

    MODEL_FILE_PREFIX = "inference"

    def __init__(self, model_dir: str, config: dict = None) -> None:
        """Initializes the BasePredictor.

        Args:
            model_dir (str): The directory where the static model files is stored.
            config (dict, optional): The configuration of model to infer. Defaults to None.
        """
        super().__init__()
        self.model_dir = Path(model_dir)
        self.config = config if config else self.load_config(self.model_dir)

        # alias predict() to the __call__()
        self.predict = self.__call__
        self.benchmark = None

    @property
    def config_path(self) -> str:
        """
        Get the path to the configuration file.

        Returns:
            str: The path to the configuration file.
        """
        return self.get_config_path(self.model_dir)

    @property
    def model_name(self) -> str:
        """
        Get the model name.

        Returns:
            str: The model name.
        """
        return self.config["Global"]["model_name"]

    @classmethod
    def get_config_path(cls, model_dir) -> str:
        """Get the path to the configuration file for the given model directory.

        Args:
            model_dir (Path): The directory where the static model files is stored.

        Returns:
            Path: The path to the configuration file.
        """
        return model_dir / f"{cls.MODEL_FILE_PREFIX}.yml"

    @classmethod
    def load_config(cls, model_dir) -> dict:
        """Load the configuration from the specified model directory.

        Args:
            model_dir (Path): The where the static model files is stored.

        Returns:
            dict: The loaded configuration dictionary.
        """
        yaml_reader = YAMLReader()
        return yaml_reader.read(cls.get_config_path(model_dir))

    @abstractmethod
    def __call__(self, input: Any, **kwargs: dict[str, Any]) -> Iterator[Any]:
        """Predict with the given input and additional keyword arguments."""
        raise NotImplementedError

    @abstractmethod
    def apply(self, input: Any) -> Iterator[Any]:
        """Predict the given input."""
        raise NotImplementedError

    @abstractmethod
    def set_predictor(self) -> None:
        """Sets up the predictor."""
        raise NotImplementedError
