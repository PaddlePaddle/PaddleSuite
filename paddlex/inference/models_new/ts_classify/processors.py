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


class GetCls:

    def __init__(self):
        super().__init__()

    def __call__(self, pred_list):
        return [self.getcls(pred) for pred in pred_list]

    def getcls(self, pred):
        pred_ts = pred[0]
        pred_ts -= np.max(pred_ts, axis=-1, keepdims=True)
        pred_ts = np.exp(pred_ts) / np.sum(np.exp(pred_ts), axis=-1, keepdims=True)
        classid = np.argmax(pred_ts, axis=-1)
        pred_score = pred_ts[classid]
        result = pd.DataFrame.from_dict({"classid": [classid], "score": [pred_score]})
        result.index.name = "sample"
        return result


class BuildPadMask:

    def __init__(self, input_data):
        super().__init__()
        self.input_data = input_data

    def __call__(self, ts_list):
        return [self.padmask(ts) for ts in ts_list]

    def padmask(self, ts):
        if "features" in self.input_data:
            ts["features"] = ts["past_target"]

        if "pad_mask" in self.input_data:
            target_dim = len(ts["features"])
            max_length = self.input_data["pad_mask"][-1]
            if max_length > 0:
                ones = np.ones(max_length, dtype=np.int32)
                if max_length != target_dim:
                    target_ndarray = np.array(ts["features"]).astype(np.float32)
                    target_ndarray_final = np.zeros(
                        [max_length, target_dim], dtype=np.int32
                    )
                    end = min(target_dim, max_length)
                    target_ndarray_final[:end, :] = target_ndarray
                    ts["features"] = target_ndarray_final
                    ones[end:] = 0.0
                    ts["pad_mask"] = ones
                else:
                    ts["pad_mask"] = ones
        return ts
