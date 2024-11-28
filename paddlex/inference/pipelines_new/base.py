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
from ...utils.subclass_register import AutoRegisterABCMetaClass
import yaml
import codecs
from pathlib import Path
from typing import Any, Dict, Optional
from ..models import create_predictor
from .components.chat_server.base import BaseChat
from .components.retriever.base import BaseRetriever
from .components.prompt_engeering.base import BaseGeneratePrompt


class BasePipeline(ABC, metaclass=AutoRegisterABCMetaClass):
    """Base Pipeline"""

    __is_base = True

    def __init__(
        self,
        device=None,
        pp_option=None,
        use_hpip: bool = False,
        hpi_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.device = device
        self.pp_option = pp_option
        self.use_hpip = use_hpip
        self.hpi_params = hpi_params

    @abstractmethod
    def predict(self, input, **kwargs):
        raise NotImplementedError("The method `predict` has not been implemented yet.")

    def create_model(self, config):

        model_dir = config["model_dir"]
        if model_dir == None:
            model_dir = config["model_name"]

        model = create_predictor(
            model_dir,
            device=self.device,
            pp_option=self.pp_option,
            use_hpip=self.use_hpip,
            hpi_params=self.hpi_params,
        )

        ########### [TODO]支持初始化传参能力
        if "batch_size" in config:
            batch_size = config["batch_size"]
            model.set_predictor(batch_size=batch_size)

        return model

    def create_pipeline(self, config):
        pipeline_name = config["pipeline_name"]
        pipeline = BasePipeline.get(pipeline_name)(
            config=config,
            device=self.device,
            pp_option=self.pp_option,
            use_hpip=self.use_hpip,
            hpi_params=self.hpi_params,
        )
        return pipeline

    def create_chat_bot(self, config):
        model_name = config["model_name"]
        chat_bot = BaseChat.get(model_name)(config)
        return chat_bot

    def create_retriever(self, config):
        model_name = config["model_name"]
        retriever = BaseRetriever.get(model_name)(config)
        return retriever

    def create_prompt_engeering(self, config):
        task_type = config["task_type"]
        pe = BaseGeneratePrompt.get(task_type)(config)
        return pe

    def __call__(self, input, **kwargs):
        return self.predict(input, **kwargs)
