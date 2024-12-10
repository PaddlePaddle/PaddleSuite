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

from typing import Any, List

import xdeploy as xd
import pandas as pd
from paddlex.inference.results import TSAdResult
from paddlex.modules.ts_anomaly_detection.model_list import MODELS

from paddlex_hpi._utils.typing import BatchData, Data
from paddlex_hpi.models.base import TSPredictor


class TSAdPredictor(TSPredictor):
    entities = MODELS

    def _build_xd_model(
        self, option: xd.RuntimeOption
    ) -> xd.ts.anomalydetection.PyOnlyAnomalyDetectionModel:
        model = xd.ts.anomalydetection.PyOnlyAnomalyDetectionModel(
            str(self.model_path),
            str(self.params_path),
            str(self.config_path),
            runtime_option=option,
        )
        return model

    def _predict(self, batch_data: BatchData) -> BatchData:
        ts_data = [data["ts"] for data in batch_data]
        xd_results = self._xd_model.batch_predict(ts_data)
        results: BatchData = []
        for data, xd_result in zip(batch_data, xd_results):
            ts_ad_result = self._create_ts_ad_result(data, xd_result)
            results.append({"result": ts_ad_result})
        return results

    def _create_ts_ad_result(self, data: Data, xd_result: Any) -> TSAdResult:
        data_dict = {
            xd_result.col_names[i]: xd_result.data[i]
            for i in range(len(xd_result.col_names))
        }
        anomaly = pd.DataFrame.from_dict(data_dict)
        anomaly.index = xd_result.dates
        anomaly.index.name = "timestamp"
        dic = {"input_path": data["input_path"], "anomaly": anomaly}
        return TSAdResult(dic)
