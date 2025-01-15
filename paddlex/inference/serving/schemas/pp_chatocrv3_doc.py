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

from ..infra.models import DataInfo, PrimaryOperations
from .shared import ocr

__all__ = [
    "ANALYZE_IMAGES_ENDPOINT",
    "InferenceParams",
    "AnalyzeImagesRequest",
    "VisualResult",
    "AnalyzeImagesResult",
    "BUILD_VECTOR_STORE_ENDPOINT",
    "BuildVectorStoreRequest",
    "BuildVectorStoreResult",
    "CHAT_ENDPOINT",
    "ChatRequest",
    "ChatResult",
    "PRIMARY_OPERATIONS",
]

ANALYZE_IMAGES_ENDPOINT: Final[str] = "/chatocr-visual"


class InferenceParams(BaseModel):
    maxLongSide: Optional[Annotated[int, Field(gt=0)]] = None


class AnalyzeImagesRequest(ocr.BaseInferRequest):
    useGeneralOcr: Optional[bool] = None
    useSealRecognition: Optional[bool] = None
    useTableRecognition: Optional[bool] = None
    inferenceParams: Optional[InferenceParams] = None


class VisualResult(BaseModel):
    texts: List[ocr.Text]
    tables: List[ocr.Table]
    inputImage: Optional[str] = None
    layoutImage: Optional[str] = None
    ocrImage: Optional[str] = None


class AnalyzeImagesResult(BaseModel):
    visualResults: List[VisualResult]
    visualInfo: dict
    dataInfo: DataInfo


BUILD_VECTOR_STORE_ENDPOINT: Final[str] = "/chatocr-vector"


class BuildVectorStoreRequest(BaseModel):
    visualInfo: dict
    minCharacters: Optional[int] = None
    llmRequestInterval: Optional[float] = None


class BuildVectorStoreResult(BaseModel):
    vectorInfo: dict


CHAT_ENDPOINT: Final[str] = "/chatocr-chat"


class ChatRequest(BaseModel):
    keyList: List[str]
    visualInfo: dict
    useVectorRetrieval: Optional[bool] = None
    vectorInfo: Optional[str] = None
    minCharacters: Optional[int] = None
    textTaskDescription: Optional[str] = None
    textOutputFormat: Optional[str] = None
    # Is the "Str" in the name unnecessary? Keep the names consistent with the
    # parameters of the wrapped function though.
    textRulesStr: Optional[str] = None
    # Should this be just "text" instead of "text content", given that there is
    # no container?
    textFewShotDemoTextContent: Optional[str] = None
    textFewShotDemoKeyValueList: Optional[str] = None
    tableTaskDescription: Optional[str] = None
    tableOutputFormat: Optional[str] = None
    tableRulesStr: Optional[str] = None
    tableFewShotDemoTextContent: Optional[str] = None
    tableFewShotDemoKeyValueList: Optional[str] = None


class ChatResult(BaseModel):
    chatResult: dict


PRIMARY_OPERATIONS: Final[PrimaryOperations] = {
    "analyzeImages": (
        ANALYZE_IMAGES_ENDPOINT,
        AnalyzeImagesRequest,
        AnalyzeImagesResult,
    ),
    "buildVectorStore": (
        BUILD_VECTOR_STORE_ENDPOINT,
        BuildVectorStoreRequest,
        BuildVectorStoreResult,
    ),
    "chat": (CHAT_ENDPOINT, ChatRequest, ChatResult),
}
