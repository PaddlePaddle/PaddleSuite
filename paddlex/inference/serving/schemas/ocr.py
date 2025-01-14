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

from typing import Final, List, Literal, Optional

from pydantic import BaseModel, Field
from typing_extensions import Annotated

from ..infra.models import DataInfo, MainOperations
from .shared import ocr

__all__ = [
    "INFER_ENDPOINT",
    "InferenceParams",
    "InferRequest",
    "OCRResult",
    "InferResult",
    "MAIN_OPERATIONS",
]

INFER_ENDPOINT: Final[str] = "/ocr"


class InferenceParams(BaseModel):
    textDetLimitSideLen: Optional[Annotated[int, Field(gt=0)]] = None
    textDetLimitType: Optional[Literal[""]] = None
    # Better to use "threshold"? Be consistent with the pipeline API though.
    textDetThresh: Optional[float] = None
    textDetBoxThresh: Optional[float] = None
    textDetMaxCandidates: Optional[float] = None
    textDetUnclipRatio: Optional[float] = None
    textDetUseDilation: Optional[bool] = None
    textRecScoreThresh: Optional[float] = None


class InferRequest(ocr.BaseInferRequest):
    useTextLineOrientation: Optional[bool] = False
    inferenceParams: Optional[InferenceParams] = None


class OCRResult(BaseModel):
    texts: List[ocr.Text]
    image: Optional[str] = None


class InferResult(BaseModel):
    ocrResults: List[OCRResult]
    dataInfo: DataInfo


MAIN_OPERATIONS: Final[MainOperations] = {
    "infer": (INFER_ENDPOINT, InferRequest, InferResult),
}
