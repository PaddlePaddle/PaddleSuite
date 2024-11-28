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

from .....utils import logging
from .base import BaseChat
import erniebot


class ErnieBotChat(BaseChat):
    """Ernie Bot Chat"""

    entities = [
        "ernie-4.0",
        "ernie-3.5",
        "ernie-3.5-8k",
        "ernie-lite",
        "ernie-tiny-8k",
        "ernie-speed",
        "ernie-speed-128k",
        "ernie-char-8k",
    ]

    def __init__(self, config):
        super().__init__()
        model_name = config.get("model_name", None)
        api_type = config.get("api_type", None)
        ak = config.get("ak", None)
        sk = config.get("sk", None)
        access_token = config.get("access_token", None)

        if model_name not in self.entities:
            raise ValueError(f"model_name must be in {self.entities} of ErnieBotChat.")

        if api_type not in ["aistudio", "qianfan"]:
            raise ValueError("api_type must be one of ['aistudio', 'qianfan']")

        if api_type == "aistudio" and access_token is None:
            raise ValueError("access_token cannot be empty when api_type is aistudio.")

        if api_type == "qianfan" and (ak is None or sk is None):
            raise ValueError("ak and sk cannot be empty when api_type is qianfan.")

        self.model_name = model_name
        self.config = config

    def generate_chat_results(self, prompt, temperature=0.001, max_retries=1):
        """
        args:
        return:
        """
        try:
            cur_config = {
                "api_type": self.config["api_type"],
                "max_retries": max_retries,
            }
            if self.config["api_type"] == "aistudio":
                cur_config["access_token"] = self.config["access_token"]
            elif self.config["api_type"] == "qianfan":
                cur_config["ak"] = self.config["ak"]
                cur_config["sk"] = self.config["sk"]
            chat_completion = erniebot.ChatCompletion.create(
                _config_=cur_config,
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=float(temperature),
            )
            llm_result = chat_completion.get_result()
            return llm_result
        except Exception as e:
            if len(e.args) < 1:
                self.ERROR_MASSAGE = "暂无权限访问ErnieBot服务，请检查访问令牌。"
            elif (
                e.args[-1]
                == "暂无权限使用，请在 AI Studio 正确获取访问令牌(access token)使用"
            ):
                self.ERROR_MASSAGE = "暂无权限访问ErnieBot服务，请检查访问令牌。"
            else:
                logging.error(e)
                self.ERROR_MASSAGE = "大模型调用失败"
        return None
