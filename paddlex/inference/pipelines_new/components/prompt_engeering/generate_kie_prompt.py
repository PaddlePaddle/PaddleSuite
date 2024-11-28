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

from .base import BaseGeneratePrompt


class GenerateKIEPrompt(BaseGeneratePrompt):
    """Generate KIE Prompt"""

    entities = ["text_kie_prompt", "table_kie_prompt"]

    def __init__(self, config):
        super().__init__()

        task_type = config.get("task_type", "")
        task_description = config.get("task_description", "")
        output_format = config.get("output_format", "")
        rules_str = config.get("rules_str", "")
        few_shot_demo_text_content = config.get("few_shot_demo_text_content", "")
        few_shot_demo_key_value_list = config.get("few_shot_demo_key_value_list", "")

        if task_description is None:
            task_description = ""

        if output_format is None:
            output_format = ""

        if rules_str is None:
            rules_str = ""

        if few_shot_demo_text_content is None:
            few_shot_demo_text_content = ""

        if few_shot_demo_key_value_list is None:
            few_shot_demo_key_value_list = ""

        if task_type not in self.entities:
            raise ValueError(
                f"task type must be in {self.entities} of GenerateKIEPrompt."
            )

        self.task_type = task_type
        self.task_description = task_description
        self.output_format = output_format
        self.rules_str = rules_str
        self.few_shot_demo_text_content = few_shot_demo_text_content
        self.few_shot_demo_key_value_list = few_shot_demo_key_value_list

    def generate_prompt(
        self,
        text_content,
        key_list,
        task_description=None,
        output_format=None,
        rules_str=None,
        few_shot_demo_text_content=None,
        few_shot_demo_key_value_list=None,
    ):
        """
        args:
        return:
        """

        if task_description is None:
            task_description = self.task_description

        if output_format is None:
            output_format = self.output_format

        if rules_str is None:
            rules_str = self.rules_str

        if few_shot_demo_text_content is None:
            few_shot_demo_text_content = self.few_shot_demo_text_content

        if few_shot_demo_key_value_list is None:
            few_shot_demo_key_value_list = self.few_shot_demo_key_value_list

        prompt = f"""{task_description}{output_format}{rules_str}{few_shot_demo_text_content}{few_shot_demo_key_value_list}"""
        if self.task_type == "table_kie_prompt":
            prompt += f"""\n结合上面，下面正式开始：\
                表格内容：```{text_content}```\
                关键词列表：{key_list}。""".replace(
                "    ", ""
            )
        elif self.task_type == "text_kie_prompt":
            prompt += f"""\n结合上面的例子，下面正式开始：\
                OCR文字：```{text_content}```\
                关键词列表：{key_list}。""".replace(
                "    ", ""
            )
        else:
            raise ValueError(f"{self.task_type} is currently not supported.")
        return prompt
