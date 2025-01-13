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

from typing import Final

from pydantic import BaseModel

from .._models import MainOperations

__all__ = ["INFER_ENDPOINT", "InferRequest", "InferResult", "MAIN_OPERATIONS"]

INFER_ENDPOINT: Final[str] = "/time-series-classification"


class InferRequest(BaseModel):
    csv: str


class InferResult(BaseModel):
    label: str
    score: float


MAIN_OPERATIONS: Final[MainOperations] = {
    "infer": (INFER_ENDPOINT, InferRequest, InferResult),
}
