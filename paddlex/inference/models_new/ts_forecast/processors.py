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

import joblib
import numpy as np
import pandas as pd


class TSDeNormalize:

    def __init__(self, scale_path, params_info):
        super().__init__()
        self.scaler = joblib.load(scale_path)
        self.params_info = params_info

    def __call__(self, preds_list):
        """apply"""
        return [self.tsdenorm(pred) for pred in preds_list]

    def tsdenorm(self, pred):
        scale_cols = pred.columns.values.tolist()
        pred[scale_cols] = self.scaler.inverse_transform(pred[scale_cols])
        return pred


class ArraytoTS:

    def __init__(self, info_params):
        super().__init__()
        self.info_params = info_params

    def __call__(self, ori_ts_list, pred_list):
        return [
            self.arraytots(ori_ts, pred) for ori_ts, pred in zip(ori_ts_list, pred_list)
        ]

    def arraytots(self, ori_ts, pred):
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
        if isinstance(self.info_params["freq"], str):
            past_target_index = ts.index
            if past_target_index.freq is None:
                past_target_index.freq = pd.infer_freq(ts.index)
            future_target_index = pd.date_range(
                past_target_index[-1] + past_target_index.freq,
                periods=pred.shape[0],
                freq=self.info_params["freq"],
                name=self.info_params["time_col"],
            )
        elif isinstance(self.info_params["freq"], int):
            start_idx = max(ts.index) + 1
            stop_idx = start_idx + pred.shape[0]
            future_target_index = pd.RangeIndex(
                start=start_idx,
                stop=stop_idx,
                step=self.info_params["freq"],
                name=self.info_params["time_col"],
            )

        future_target = pd.DataFrame(
            np.reshape(pred, newshape=[pred.shape[0], -1]),
            index=future_target_index,
            columns=column_name,
        )
        return future_target
