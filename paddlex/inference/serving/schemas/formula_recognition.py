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

from pydantic import BaseModel, Field
from typing_extensions import Annotated

from .._models import DataInfo, MainOperations
from .shared import ocr

__all__ = [
    "INFER_ENDPOINT",
    "InferenceParams",
    "InferRequest",
    "FormulaRecResult",
    "InferResult",
    "MAIN_OPERATIONS",
]

INFER_ENDPOINT: Final[str] = "/formula-recognition"


class InferenceParams(BaseModel):
    maxLongSide: Optional[Annotated[int, Field(gt=0)]] = None


class InferRequest(ocr.BaseInferRequest):
    useTextLineOrientation: Optional[bool] = False
    inferenceParams: Optional[InferenceParams] = None


class FormulaRecResult(BaseModel):
    formulas: List[ocr.Formula]
    inputImage: Optional[str] = None
    layoutImage: Optional[str] = None
    ocrImage: Optional[str] = None


class InferResult(BaseModel):
    formulaRecResults: List[FormulaRecResult]
    dataInfo: DataInfo


MAIN_OPERATIONS: Final[MainOperations] = {
    "infer": (INFER_ENDPOINT, InferRequest, InferResult),
}
