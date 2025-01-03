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

from ._common import cv as cv_common, ocr as ocr_common
from ....utils import logging
from .. import utils as serving_utils
from ..app import AppConfig, create_app
from ..models import NoResultResponse, ResultResponse, DataInfo


InferRequest: Type[ocr_common.InferRequest] = ocr_common.InferRequest


Point: TypeAlias = Annotated[List[int], Field(min_length=2, max_length=2)]
Polygon: TypeAlias = Annotated[List[Point], Field(min_length=3)]


class Text(BaseModel):
    poly: Polygon
    text: str
    score: float


class OCRResult(BaseModel):
    texts: List[Text]
    image: str


class InferResult(BaseModel):
    ocrResults: List[OCRResult]
    dataInfo: DataInfo


def create_pipeline_app(pipeline: Any, app_config: AppConfig) -> FastAPI:
    app, ctx = create_app(
        pipeline=pipeline, app_config=app_config, app_aiohttp_session=True
    )

    ocr_common.update_app_context(ctx)

    @app.post(
        "/ocr",
        operation_id="infer",
        responses={422: {"model": NoResultResponse}},
        response_model_exclude_none=True,
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

        try:
            result = await pipeline.infer(images)

            ocr_results: List[OCRResult] = []
            for i, item in enumerate(result):
                texts: List[Text] = []
                for poly, text, score in zip(
                    item["dt_polys"], item["rec_text"], item["rec_score"]
                ):
                    texts.append(Text(poly=poly, text=text, score=score))
                image = await serving_utils.call_async(
                    cv_common.postprocess_image,
                    item.img,
                    log_id=log_id,
                    filename=f"image_{i}.jpg",
                    file_storage=ctx.extra["file_storage"],
                    return_url=ctx.extra["return_img_urls"],
                    max_img_size=ctx.extra["max_output_img_size"],
                )
                ocr_results.append(OCRResult(texts=texts, image=image))

            return ResultResponse[InferResult](
                logId=log_id,
                result=InferResult(
                    ocrResults=ocr_results,
                    dataInfo=data_info,
                ),
            )

        except Exception:
            logging.exception("Unexpected exception")
            raise HTTPException(status_code=500, detail="Internal server error")

    return app
