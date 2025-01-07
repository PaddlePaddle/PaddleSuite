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

from typing import Any, List, Type

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypeAlias

from .. import _utils as serving_utils
from .._app import AppConfig, create_app, main_operation
from .._models import DataInfo, ResultResponse
from ._common import ocr as ocr_common

InferRequest: Type[ocr_common.InferRequest] = ocr_common.InferRequest

Point: TypeAlias = Annotated[List[int], Field(min_length=2, max_length=2)]
BoundingBox: TypeAlias = Annotated[List[float], Field(min_length=4, max_length=4)]


class Table(BaseModel):
    bbox: BoundingBox
    html: str


class TableRecResult(BaseModel):
    tables: List[Table]
    inputImage: str
    layoutImage: str
    ocrImage: str


class InferResult(BaseModel):
    tableRecResults: List[TableRecResult]
    dataInfo: DataInfo


def create_pipeline_app(pipeline: Any, app_config: AppConfig) -> FastAPI:
    app, ctx = create_app(
        pipeline=pipeline, app_config=app_config, app_aiohttp_session=True
    )

    ocr_common.update_app_context(ctx)

    @main_operation(
        app,
        "/table-recognition",
        "infer",
    )
    async def _infer(request: InferRequest) -> ResultResponse[InferResult]:
        pipeline = ctx.pipeline

        log_id = serving_utils.generate_log_id()

        if request.inferenceParams:
            max_long_side = request.inferenceParams.maxLongSide
            if max_long_side:
                raise HTTPException(
                    status_code=422,
                    detail="`max_long_side` is currently not supported.",
                )

        images, data_info = await ocr_common.get_images(request, ctx)

        result = await pipeline.infer(images)

        table_rec_results: List[TableRecResult] = []
        for i, (img, item) in enumerate(zip(images, result)):
            tables: List[Table] = []
            for subitem in item["table_result"]:
                tables.append(
                    Table(
                        bbox=subitem["layout_bbox"],
                        html=subitem["html"],
                    )
                )
            input_img, layout_img, ocr_img = await ocr_common.postprocess_images(
                log_id=log_id,
                index=i,
                app_context=ctx,
                input_image=img,
                layout_image=item["layout_result"].img,
                ocr_image=item["ocr_result"].img,
            )
            table_rec_results.append(
                TableRecResult(
                    tables=tables,
                    inputImage=input_img,
                    layoutImage=layout_img,
                    ocrImage=ocr_img,
                )
            )

        return ResultResponse[InferResult](
            logId=serving_utils.generate_log_id(),
            result=InferResult(
                tableRecResults=table_rec_results,
                dataInfo=data_info,
            ),
        )

    return app
