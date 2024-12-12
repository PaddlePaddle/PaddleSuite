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
import pandas as pd


class GetAnomaly:

    def __init__(self, model_threshold, info_params):
        super().__init__()
        self.model_threshold = model_threshold
        self.info_params = info_params

    def __call__(self, ori_ts_list, pred_list):
        return [
            self.getanomaly(ori_ts, pred)
            for ori_ts, pred in zip(ori_ts_list, pred_list)
        ]

    def getanomaly(self, ori_ts, pred):
        pred = pred[0]
        if ori_ts.get("past_target", None) is not None:
            ts = ori_ts["past_target"]
        elif ori_ts.get("observed_cov_numeric", None) is not None:
            ts = ori_ts["observed_cov_numeric"]
        elif ori_ts.get("known_cov_numeric", None) is not None:
            ts = ori_ts["known_cov_numeric"]
        elif ori_ts.get("static_cov_numeric", None) is not None:
            ts = ori_ts["static_cov_numeric"]
        else:
            raise ValueError("No value in ori_ts")
        column_name = (
            self.info_params["target_cols"]
            if "target_cols" in self.info_params
            else self.info_params["feature_cols"]
        )

        anomaly_score = np.mean(np.square(pred - np.array(ts)), axis=-1)
        anomaly_label = (anomaly_score >= self.model_threshold) + 0

        past_target_index = ts.index
        past_target_index.name = self.info_params["time_col"]
        anomaly_label = pd.DataFrame(
            np.reshape(anomaly_label, newshape=[pred.shape[0], -1]),
            index=past_target_index,
            columns=["label"],
        )
        return anomaly_label
