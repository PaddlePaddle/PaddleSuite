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

from typing import Generic, TypeVar, Union, Literal, List, Optional

from pydantic import BaseModel


class NoResultResponse(BaseModel):
    logId: str
    errorCode: int
    errorMsg: str


ResultT = TypeVar("ResultT", bound=BaseModel)


class ResultResponse(BaseModel, Generic[ResultT]):
    logId: str
    errorCode: Literal[0] = 0
    errorMsg: Literal["Success"] = "Success"
    result: ResultT


Response = Union[ResultResponse, NoResultResponse]


class ImageInfo(BaseModel):
    width: int
    height: int


class PDFPageInfo(BaseModel):
    width: int
    height: int


class PDFInfo(BaseModel):
    numPages: int
    pages: List[PDFPageInfo]


class DataInfo(BaseModel):
    image: Optional[ImageInfo] = None
    pdf: Optional[PDFInfo] = None

    # TODO: Validate that only one field is set
