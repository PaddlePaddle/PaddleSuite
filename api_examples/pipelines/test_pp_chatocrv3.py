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

from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="PP-ChatOCRv3-doc")

img_path = "./test_samples/vehicle_certificate-1.png"
key_list = ["驾驶室准乘人数"]

visual_predict_res = pipeline.visual_predict(
    img_path,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_common_ocr=True,
    use_seal_recognition=True,
    use_table_recognition=True,
)

# ####[TODO] 增加类别信息
visual_info_list = []
for res in visual_predict_res:
    # res['layout_parsing_result'].save_results("./output/")
    # print(res["visual_info"])
    visual_info_list.append(res["visual_info"])

#pipeline.save_visual_info_list(visual_info_list, "./res_visual_info/tmp_visual_info.json")

#visual_info_list = pipeline.load_visual_info_list("./res_visual_info/tmp_visual_info.json")

vector_info = pipeline.build_vector(visual_info_list)

#pipeline.save_vector(vector_info, "./res_visual_info/tmp_vector_info.json")

#vector_info = pipeline.load_vector("./res_visual_info/tmp_vector_info.json")

chat_result = pipeline.chat(key_list, visual_info_list, vector_info=vector_info)

print(chat_result)
