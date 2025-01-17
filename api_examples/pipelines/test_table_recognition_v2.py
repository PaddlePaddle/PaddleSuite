# # copyright (c) 2024 PaddlePaddle Authors. All Rights Reserve.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #    http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

# from paddlex import create_pipeline

# # pipeline = create_pipeline(pipeline="table_recognition_v2")
# # pipeline = create_pipeline(pipeline="table_recognition")
# pipeline = create_pipeline(pipeline="/workspace/jieru/PaddleX/paddlex/configs/pipelines/table_recognition.yaml")

# # output = pipeline("./test_samples/table_recognition.jpg")
# output = pipeline("/workspace/jieru/51.png")

# for res in output:
#     print(res)
#     # res.save_results("/workspace/jieru/pipe_out/")


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

pipeline = create_pipeline(pipeline="table_recognition_v2")

output = pipeline.predict("./test_samples/table_recognition.jpg")

for res in output:
    res.print()
    res.save_to_img("/workspace/jieru/pipe_out")
    res.save_to_json("/workspace/jieru/pipe_out")
    res.save_to_xlsx("/workspace/jieru/pipe_out")
    res.save_to_html("/workspace/jieru/pipe_out")
