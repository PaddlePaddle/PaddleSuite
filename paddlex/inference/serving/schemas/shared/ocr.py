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

from typing import List, Optional

from pydantic import BaseModel, Field
from typing_extensions import Annotated, Literal, TypeAlias

FileType: TypeAlias = Literal[0, 1]


class BaseInferRequest(BaseModel):
    file: str
    fileType: Optional[FileType] = None
    # Should it be "Classification" instead of "Classify"? Keep the names
    # consistent with the parameters of the wrapped function though.
    useDocOrientationClassify: Optional[bool] = None
    useDocUnwarping: Optional[bool] = None


Point: TypeAlias = Annotated[List[int], Field(min_length=2, max_length=2)]
Polygon: TypeAlias = Annotated[List[Point], Field(min_length=3)]


class Text(BaseModel):
    poly: Polygon
    text: str
    score: float


BoundingBox: TypeAlias = Annotated[List[float], Field(min_length=4, max_length=4)]


class Table(BaseModel):
    bbox: BoundingBox
    html: str


class Formula(BaseModel):
    poly: Polygon
    latex: str
