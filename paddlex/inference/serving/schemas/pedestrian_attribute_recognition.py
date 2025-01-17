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

from ..infra.models import PrimaryOperations
from .shared import object_detection

__all__ = [
    "INFER_ENDPOINT",
    "InferenceParams",
    "InferRequest",
    "Attribute",
    "Pedestrian",
    "InferResult",
    "PRIMARY_OPERATIONS",
]

INFER_ENDPOINT: Final[str] = "/pedestrian-attribute-recognition"


class InferenceParams(BaseModel):
    detThreshold: Optional[float] = None
    clsThreshold: Optional[float] = None


class InferRequest(BaseModel):
    image: str
    inferenceParams: Optional[InferenceParams] = None


class Attribute(BaseModel):
    label: str
    score: float


class Pedestrian(BaseModel):
    bbox: object_detection.BoundingBox
    attributes: List[Attribute]
    score: float


class InferResult(BaseModel):
    pedestrians: List[Pedestrian]
    image: Optional[str] = None


PRIMARY_OPERATIONS: Final[PrimaryOperations] = {
    "infer": (INFER_ENDPOINT, InferRequest, InferResult),
}
