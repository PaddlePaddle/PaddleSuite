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

from ..base import BasePipeline

from typing import Any, Dict, Optional

# import numpy as np
# import cv2
from .result import VisualInfoResult
import re

########## [TODO]后续需要更新路径
from ...components.transforms import ReadImage

import json

from ....utils import logging


class PP_ChatOCRv3_doc_Pipeline(BasePipeline):
    """PP-ChatOCRv3-doc Pipeline"""

    entities = "PP-ChatOCRv3-doc"

    def __init__(
        self,
        config,
        device=None,
        pp_option=None,
        use_hpip: bool = False,
        hpi_params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            device=device, pp_option=pp_option, use_hpip=use_hpip, hpi_params=hpi_params
        )

        self.inintial_predictor(config)

        self.img_reader = ReadImage(format="BGR")

    def inintial_predictor(self, config):
        # layout_parsing_config = config['SubPipelines']["LayoutParser"]
        # self.layout_parsing_pipeline = self.create_pipeline(layout_parsing_config)

        chat_bot_config = config["SubModules"]["LLM_Chat"]
        self.chat_bot = self.create_chat_bot(chat_bot_config)

        retriever_config = config["SubModules"]["LLM_Retriever"]
        self.retriever = self.create_retriever(retriever_config)

        text_pe_config = config["SubModules"]["PromptEngneering"]["KIE_CommonText"]
        self.text_pe = self.create_prompt_engeering(text_pe_config)

        table_pe_config = config["SubModules"]["PromptEngneering"]["KIE_Table"]
        self.table_pe = self.create_prompt_engeering(table_pe_config)

        return

    def decode_visual_result(self, layout_parsing_result):
        text_paragraphs_ocr_res = layout_parsing_result["text_paragraphs_ocr_res"]
        seal_res_list = layout_parsing_result["seal_res_list"]
        normal_text_dict = {}
        layout_type = "text"
        for text in text_paragraphs_ocr_res["rec_text"]:
            if layout_type not in normal_text_dict:
                normal_text_dict[layout_type] = text
            else:
                normal_text_dict[layout_type] += f"\n {text}"

        layout_type = "seal"
        for seal_res in seal_res_list:
            for text in seal_res["rec_text"]:
                if layout_type not in normal_text_dict:
                    normal_text_dict[layout_type] = text
                else:
                    normal_text_dict[layout_type] += f"\n {text}"

        table_res_list = layout_parsing_result["table_res_list"]
        table_text_list = []
        table_html_list = []
        for table_res in table_res_list:
            table_html_list.append(table_res["pred_html"])
            single_table_text = " ".join(table_res["table_ocr_pred"]["rec_text"])
            table_text_list.append(single_table_text)

        visual_info = {}
        visual_info["normal_text_dict"] = normal_text_dict
        visual_info["table_text_list"] = table_text_list
        visual_info["table_html_list"] = table_html_list
        return VisualInfoResult(visual_info)

    def visual_predict(
        self,
        input,
        use_doc_orientation_classify=True,
        use_doc_unwarping=True,
        use_common_ocr=True,
        use_seal_recognition=True,
        use_table_recognition=True,
        **kwargs,
    ):

        if not isinstance(input, list):
            input_list = [input]
        else:
            input_list = input

        img_id = 1
        for input in input_list:
            if isinstance(input, str):
                image_array = next(self.img_reader(input))[0]["img"]
            else:
                image_array = input

            assert len(image_array.shape) == 3

            layout_parsing_result = next(
                self.layout_parsing_pipeline.predict(
                    image_array,
                    use_doc_orientation_classify=use_doc_orientation_classify,
                    use_doc_unwarping=use_doc_unwarping,
                    use_common_ocr=use_common_ocr,
                    use_seal_recognition=use_seal_recognition,
                    use_table_recognition=use_table_recognition,
                )
            )

            visual_info = self.decode_visual_result(layout_parsing_result)

            visual_predict_res = {
                "layout_parsing_result": layout_parsing_result,
                "visual_info": visual_info,
            }
            yield visual_predict_res

    def save_visual_info_list(self, visual_info, save_path):
        if not isinstance(visual_info, list):
            visual_info_list = [visual_info]
        else:
            visual_info_list = visual_info

        with open(save_path, "w") as fout:
            fout.write(json.dumps(visual_info_list, ensure_ascii=False) + "\n")
        return

    def load_visual_info_list(self, data_path):
        with open(data_path, "r") as fin:
            data = fin.readline()
            visual_info_list = json.loads(data)
        return visual_info_list

    def merge_visual_info_list(self, visual_info_list):
        all_normal_text_list = []
        all_table_text_list = []
        all_table_html_list = []
        for single_visual_info in visual_info_list:
            normal_text_dict = single_visual_info["normal_text_dict"]
            table_text_list = single_visual_info["table_text_list"]
            table_html_list = single_visual_info["table_html_list"]
            all_normal_text_list.append(normal_text_dict)
            all_table_text_list.extend(table_text_list)
            all_table_html_list.extend(table_html_list)
        return all_normal_text_list, all_table_text_list, all_table_html_list

    def build_vector(self, visual_info, min_characters=3500, llm_request_interval=1.0):

        if not isinstance(visual_info, list):
            visual_info_list = [visual_info]
        else:
            visual_info_list = visual_info

        all_visual_info = self.merge_visual_info_list(visual_info_list)
        all_normal_text_list, all_table_text_list, all_table_html_list = all_visual_info

        all_normal_text_str = "".join(
            ["\n".join(e.values()) for e in all_normal_text_list]
        )
        vector_info = {}

        all_items = []
        for i, normal_text_dict in enumerate(all_normal_text_list):
            for type, text in normal_text_dict.items():
                all_items += [f"{type}：{text}"]

        if len(all_normal_text_str) > min_characters:
            vector_info["flag_too_short_text"] = False
            vector_info["vector"] = self.retriever.generate_vector_database(all_items)
        else:
            vector_info["flag_too_short_text"] = True
            vector_info["vector"] = all_items
        return vector_info

    def format_key(self, key_list):
        """format key"""
        if key_list == "":
            return []

        if isinstance(key_list, list):
            return key_list

        if isinstance(key_list, str):
            key_list = re.sub(r"[\t\n\r\f\v]", "", key_list)
            key_list = key_list.replace("，", ",").split(",")
            return key_list

        return []

    def fix_llm_result_format(self, llm_result):
        if not llm_result:
            return {}

        if "json" in llm_result or "```" in llm_result:
            llm_result = (
                llm_result.replace("```", "").replace("json", "").replace("/n", "")
            )
            llm_result = llm_result.replace("[", "").replace("]", "")

        try:
            llm_result = json.loads(llm_result)
            llm_result_final = {}
            for key in llm_result:
                value = llm_result[key]
                if isinstance(value, list):
                    if len(value) > 0:
                        llm_result_final[key] = value[0]
                else:
                    llm_result_final[key] = value
            return llm_result_final

        except:
            results = (
                llm_result.replace("\n", "")
                .replace("    ", "")
                .replace("{", "")
                .replace("}", "")
            )
            if not results.endswith('"'):
                results = results + '"'
            pattern = r'"(.*?)": "([^"]*)"'
            matches = re.findall(pattern, str(results))
            if len(matches) > 0:
                llm_result = {k: v for k, v in matches}
                return llm_result
            else:
                return {}

    def generate_and_merge_chat_results(
        self, prompt, key_list, final_results, failed_results
    ):

        llm_result = self.chat_bot.generate_chat_results(prompt)
        llm_result = self.fix_llm_result_format(llm_result)

        for key, value in llm_result.items():
            if value not in failed_results and key in key_list:
                key_list.remove(key)
                final_results[key] = value
        return

    def chat(
        self,
        visual_info,
        key_list,
        vector_info,
        text_task_description=None,
        text_output_format=None,
        text_rules_str=None,
        text_few_shot_demo_text_content=None,
        text_few_shot_demo_key_value_list=None,
        table_task_description=None,
        table_output_format=None,
        table_rules_str=None,
        table_few_shot_demo_text_content=None,
        table_few_shot_demo_key_value_list=None,
    ):

        key_list = self.format_key(key_list)
        if len(key_list) == 0:
            return {"chat_res": "输入的key_list无效！"}

        if not isinstance(visual_info, list):
            visual_info_list = [visual_info]
        else:
            visual_info_list = visual_info

        all_visual_info = self.merge_visual_info_list(visual_info_list)
        all_normal_text_list, all_table_text_list, all_table_html_list = all_visual_info

        final_results = {}
        failed_results = ["大模型调用失败", "未知", "未找到关键信息", "None", ""]

        for all_table_info in [all_table_html_list, all_table_text_list]:
            for table_info in all_table_info:
                if len(key_list) == 0:
                    continue

                prompt = self.table_pe.generate_prompt(
                    table_info,
                    key_list,
                    task_description=table_task_description,
                    output_format=table_output_format,
                    rules_str=table_rules_str,
                    few_shot_demo_text_content=table_few_shot_demo_text_content,
                    few_shot_demo_key_value_list=table_few_shot_demo_key_value_list,
                )

                self.generate_and_merge_chat_results(
                    prompt, key_list, final_results, failed_results
                )

        if len(key_list) > 0:
            question_key_list = [f"抽取关键信息:{key}" for key in key_list]
            vector = vector_info["vector"]
            if not vector_info["flag_too_short_text"]:
                related_text = self.retriever.similarity_retrieval(
                    question_key_list, vector
                )
            else:
                related_text = " ".join(vector)

            prompt = self.text_pe.generate_prompt(
                related_text,
                key_list,
                task_description=text_task_description,
                output_format=text_output_format,
                rules_str=text_rules_str,
                few_shot_demo_text_content=text_few_shot_demo_text_content,
                few_shot_demo_key_value_list=text_few_shot_demo_key_value_list,
            )

            self.generate_and_merge_chat_results(
                prompt, key_list, final_results, failed_results
            )

        return final_results

    def predict(self, *args, **kwargs):
        logging.error(
            "PP-ChatOCRv3-doc Pipeline do not support to call `predict()` directly! Please invoke `visual_predict`, `build_vector`, `chat` sequentially to obtain the result."
        )
        return
