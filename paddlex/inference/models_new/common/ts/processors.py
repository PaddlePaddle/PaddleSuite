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

from pathlib import Path
from copy import deepcopy
import joblib
import numpy as np
import pandas as pd

from .funcs import load_from_dataframe, time_feature


__all__ = [
    "BuildTSDataset",
    "TSCutOff",
    "TSNormalize",
    "TimeFeature",
    "TStoArray",
    "TStoBatch",
]


class TSCutOff:

    def __init__(self, size):
        super().__init__()
        self.size = size

    def __call__(self, ts_list):
        return [self.cutoff(ts) for ts in ts_list]

    def cutoff(self, ts):
        skip_len = self.size.get("skip_chunk_len", 0)
        if len(ts) < self.size["in_chunk_len"] + skip_len:
            raise ValueError(
                f"The length of the input data is {len(ts)}, but it should be at least {self.size['in_chunk_len'] + self.size['skip_chunk_len']} for training."
            )
        ts_data = ts[-(self.size["in_chunk_len"] + skip_len) :]
        return ts_data


class TSNormalize:

    def __init__(self, scale_path, params_info):
        super().__init__()
        self.scaler = joblib.load(scale_path)
        self.params_info = params_info

    def __call__(self, ts_list):
        return [self.tsnorm(ts) for ts in ts_list]

    def tsnorm(self, ts):
        if self.params_info.get("target_cols", None) is not None:
            ts[self.params_info["target_cols"]] = self.scaler.transform(
                ts[self.params_info["target_cols"]]
            )
        if self.params_info.get("feature_cols", None) is not None:
            ts[self.params_info["feature_cols"]] = self.scaler.transform(
                ts[self.params_info["feature_cols"]]
            )

        return ts


class BuildTSDataset:

    def __init__(self, params_info):
        super().__init__()
        self.params_info = params_info

    def __call__(self, ts_list):
        """apply"""
        return [self.buildtsdata(ts) for ts in ts_list]

    def buildtsdata(self, ts):
        """apply"""
        ts_data = load_from_dataframe(ts, **self.params_info)
        return ts_data


class TimeFeature:

    def __init__(self, params_info, size, holiday=False):
        super().__init__()
        self.freq = params_info["freq"]
        self.size = size
        self.holiday = holiday

    def __call__(self, ts_list):
        return [self.timefeat(ts) for ts in ts_list]

    def timefeat(self, ts):
        """apply"""
        if not self.holiday:
            ts = time_feature(
                ts,
                self.freq,
                ["hourofday", "dayofmonth", "dayofweek", "dayofyear"],
                self.size["out_chunk_len"],
            )
        else:
            ts = time_feature(
                ts,
                self.freq,
                [
                    "minuteofhour",
                    "hourofday",
                    "dayofmonth",
                    "dayofweek",
                    "dayofyear",
                    "monthofyear",
                    "weekofyear",
                    "holidays",
                ],
                self.size["out_chunk_len"],
            )
        return ts


class TStoArray:

    def __init__(self, input_data):
        super().__init__()
        self.input_data = input_data

    def __call__(self, ts_list):
        return [self.tstoarray(ts) for ts in ts_list]

    def tstoarray(self, ts):
        ts_list = []
        input_name = list(self.input_data.keys())
        input_name.sort()
        for key in input_name:
            ts_list.append(np.array(ts[key]).astype("float32"))

        return ts_list


class TStoBatch:
    def __call__(self, ts_list):
        n = len(ts_list[0])
        return [np.stack([ts[i] for ts in ts_list], axis=0) for i in range(n)]
