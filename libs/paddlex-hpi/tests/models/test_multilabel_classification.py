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

import numpy as np
from paddlex.inference.results import MLClassResult
from tests.models.base import BaseTestPredictor

from paddlex_hpi.models import MLClasPredictor

MODEL_URL = "https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hpi/tests/models/ml_clas_model.zip"
INPUT_DATA_URL = "https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hpi/tests/models/ml_clas_input.jpg"
EXPECTED_RESULT_URL = "https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hpi/tests/models/ml_clas_result.json"


class TestMLClasPredictor(BaseTestPredictor):
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
        return MLClasPredictor

    def _check_result(self, result, expected_result):
        assert isinstance(result, MLClassResult)
        assert set(result) == set(expected_result)
        assert result["class_ids"] == expected_result["class_ids"]
        assert np.allclose(
            np.array(result["scores"]),
            np.array(expected_result["scores"]),
            rtol=1e-2,
            atol=1e-3,
        )
        assert result["label_names"] == expected_result["label_names"]