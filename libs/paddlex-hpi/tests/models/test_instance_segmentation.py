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

from paddlex.inference.results import InstanceSegResult
from tests.models.base import BaseTestPredictor
from tests.testing_utils.cv import compare_det_results

from paddlex_hpi.models import InstanceSegPredictor

MODEL_URL = "https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hpi/tests/models/instance_seg_model.zip"
INPUT_DATA_URL = "https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hpi/tests/models/instance_seg_input.jpg"
EXPECTED_RESULT_URL = "https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hpi/tests/models/instance_seg_result.json"


class TestInstanceSegPredictor(BaseTestPredictor):
    @property
    def model_url(self):
        return MODEL_URL

    @property
    def input_data_url(self):
        return INPUT_DATA_URL

    @property
    def expected_result_url(self):
        return EXPECTED_RESULT_URL

    @property
    def predictor_cls(self):
        return InstanceSegPredictor

    def _check_result(self, result, expected_result):
        assert isinstance(result, InstanceSegResult)
        assert set(result) == set(expected_result)
        # TODO: Check masks
        compare_det_results(
            [obj["coordinate"] for obj in result["boxes"]],
            [obj["coordinate"] for obj in expected_result["boxes"]],
            labels1=[obj["cls_id"] for obj in result["boxes"]],
            labels2=[obj["cls_id"] for obj in expected_result["boxes"]],
            scores1=[obj["score"] for obj in result["boxes"]],
            scores2=[obj["score"] for obj in expected_result["boxes"]],
        )