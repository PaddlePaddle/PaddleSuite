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

import json
import shutil
import tempfile
from pathlib import Path
from types import GeneratorType

import pytest
from tests.testing_utils.download import download, download_and_extract
from tests.testing_utils.misc import get_filename

NUM_INPUT_FILES = 10
DEVICES = ["cpu", "gpu:0"]
BATCH_SIZES = [1, 2, 4]


class BaseTestPredictor(object):
    @property
    def model_dir(self):
        raise NotImplementedError

    @property
    def model_url(self):
        raise NotImplementedError

    @property
    def input_data_url(self):
        raise NotImplementedError

    @property
    def expected_result_url(self):
        raise NotImplementedError

    @property
    def predictor_cls(self):
        raise NotImplementedError

    @pytest.fixture(scope="class")
    def data_dir(self):
        with tempfile.TemporaryDirectory() as td:
            yield Path(td)

    @pytest.fixture(scope="class")
    def model_path(self, data_dir):
        download_and_extract(self.model_url, data_dir, "model")
        yield data_dir / "model"

    @pytest.fixture(scope="class")
    def input_data_path(self, data_dir):
        input_data_path = (data_dir / get_filename(self.input_data_url)).with_stem(
            "test"
        )
        download(self.input_data_url, input_data_path)
        yield input_data_path

    @pytest.fixture(scope="class")
    def input_data_dir(self, data_dir, input_data_path):
        input_data_dir = data_dir / "input_data"
        input_data_dir.mkdir()
        for i in range(NUM_INPUT_FILES):
            shutil.copy(
                input_data_path,
                (input_data_dir / f"test_{i}").with_suffix(input_data_path.suffix),
            )
        yield input_data_dir

    @pytest.fixture(scope="class")
    def expected_result(self, data_dir):
        expected_result_path = data_dir / "expected.json"
        download(self.expected_result_url, expected_result_path)
        with open(expected_result_path, "r", encoding="utf-8") as f:
            expected_result = json.load(f)
        yield expected_result

    @pytest.mark.parametrize("device", DEVICES)
    def test___call__single_input_data(
        self, model_path, input_data_path, device, expected_result
    ):
        predictor = self.predictor_cls(model_path, device=device)
        output = predictor(str(input_data_path))
        self._check_output(output, expected_result, 1)
        output = predictor([str(input_data_path), str(input_data_path)])
        self._check_output(output, expected_result, 2)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    def test___call__input_data_dir(
        self, model_path, input_data_dir, device, batch_size, expected_result
    ):
        predictor = self.predictor_cls(model_path, device=device)
        predictor.set_predictor(batch_size=batch_size)
        output = predictor(str(input_data_dir))
        self._check_output(output, expected_result, NUM_INPUT_FILES)

    def _check_output(self, output, expected_result, expected_num_results):
        assert isinstance(output, GeneratorType)
        # Note that this exhausts the generator
        output = list(output)
        assert len(output) == expected_num_results
        for result in output:
            self._check_result(result, expected_result)

    def _check_result(self, result, expected_result):
        raise NotImplementedError
