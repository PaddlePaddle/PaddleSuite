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

from typing import Dict, Final, List, Optional

from pydantic import BaseModel, Field
from typing_extensions import Annotated

from ..infra.models import DataInfo, PrimaryOperations
from .shared import ocr

__all__ = [
    "INFER_ENDPOINT",
    "InferenceParams",
    "InferRequest",
    "LayoutParsingResult",
    "InferResult",
    "PRIMARY_OPERATIONS",
]

INFER_ENDPOINT: Final[str] = "/layout-parsing"


class InferenceParams(BaseModel):
    maxLongSide: Optional[Annotated[int, Field(gt=0)]] = None


class InferRequest(ocr.BaseInferRequest):
    useGeneralOcr: Optional[bool] = None
    useSealRecognition: Optional[bool] = None
    useTableRecognition: Optional[bool] = None
    inferenceParams: Optional[InferenceParams] = None


class LayoutParsingResult(BaseModel):
    prunedResult: dict
    outputImages: Optional[Dict[str, str]] = None
    inputImage: Optional[str] = None


class InferResult(BaseModel):
    layoutParsingResults: List[LayoutParsingResult]
    dataInfo: DataInfo


PRIMARY_OPERATIONS: Final[PrimaryOperations] = {
    "infer": (INFER_ENDPOINT, InferRequest, InferResult),
}
