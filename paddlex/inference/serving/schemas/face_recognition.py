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

from pydantic import BaseModel

from .._models import MainOperations
from .shared import object_detection

__all__ = [
    "ImageLabelPair",
    "BUILD_INDEX_ENDPOINT",
    "BuildIndexRequest",
    "BuildIndexResult",
    "ADD_IMAGES_TO_INDEX_ENDPOINT",
    "AddImagesToIndexRequest",
    "AddImagesToIndexResult",
    "REMOVE_IMAGES_FROM_INDEX_ENDPOINT",
    "RemoveImagesFromIndexRequest",
    "RemoveImagesFromIndexResult",
    "INFER_ENDPOINT",
    "InferenceParams",
    "InferRequest",
    "RecResult",
    "Face",
    "InferResult",
    "MAIN_OPERATIONS",
]


class ImageLabelPair(BaseModel):
    image: str
    label: str


BUILD_INDEX_ENDPOINT: Final[str] = "/face-recognition-index-build"


class BuildIndexRequest(BaseModel):
    imageLabelPairs: List[ImageLabelPair]


class BuildIndexResult(BaseModel):
    indexKey: str
    idMap: Dict[int, str]


ADD_IMAGES_TO_INDEX_ENDPOINT: Final[str] = "/face-recognition-index-add"


class AddImagesToIndexRequest(BaseModel):
    imageLabelPairs: List[ImageLabelPair]
    indexKey: Optional[str] = None


class AddImagesToIndexResult(BaseModel):
    idMap: Dict[int, str]


REMOVE_IMAGES_FROM_INDEX_ENDPOINT: Final[str] = "/face-recognition-index-remove"


class RemoveImagesFromIndexRequest(BaseModel):
    ids: List[int]
    indexKey: Optional[str] = None


class RemoveImagesFromIndexResult(BaseModel):
    idMap: Dict[int, str]


INFER_ENDPOINT: Final[str] = "/face-recognition-infer"


class InferenceParams(BaseModel):
    detThreshold: Optional[float] = None
    recThreshold: Optional[float] = None
    topK: Optional[int] = None


class InferRequest(BaseModel):
    image: str
    indexKey: Optional[str] = None
    inferenceParams: Optional[InferenceParams] = None


class RecResult(BaseModel):
    label: str
    score: float


class Face(BaseModel):
    bbox: object_detection.BoundingBox
    recResults: List[RecResult]
    score: float


class InferResult(BaseModel):
    faces: List[Face]
    image: Optional[str] = None


MAIN_OPERATIONS: Final[MainOperations] = {
    "buildIndex": (BUILD_INDEX_ENDPOINT, BuildIndexRequest, BuildIndexResult),
    "addImagesToIndex": (
        ADD_IMAGES_TO_INDEX_ENDPOINT,
        AddImagesToIndexRequest,
        AddImagesToIndexResult,
    ),
    "removeImagesFromIndex": (
        REMOVE_IMAGES_FROM_INDEX_ENDPOINT,
        RemoveImagesFromIndexRequest,
        RemoveImagesFromIndexResult,
    ),
    "infer": (INFER_ENDPOINT, InferRequest, InferResult),
}
