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

from typing import Final, List, Optional

from pydantic import BaseModel

from ..infra.models import DataInfo, MainOperations
from .shared import ocr

__all__ = [
    "INFER_ENDPOINT",
    "InferRequest",
    "TableRecResult",
    "InferResult",
    "MAIN_OPERATIONS",
]

INFER_ENDPOINT: Final[str] = "/table-recognition"


class InferRequest(ocr.BaseInferRequest):
    pass


class TableRecResult(BaseModel):
    tables: List[ocr.Table]
    inputImage: Optional[str] = None
    layoutImage: Optional[str] = None
    ocrImage: Optional[str] = None


class InferResult(BaseModel):
    tableRecResults: List[TableRecResult]
    dataInfo: DataInfo


MAIN_OPERATIONS: Final[MainOperations] = {
    "infer": (INFER_ENDPOINT, InferRequest, InferResult),
}
