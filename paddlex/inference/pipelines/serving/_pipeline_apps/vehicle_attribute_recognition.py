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

from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypeAlias

from .....utils import logging
from ...attribute_recognition import VehicleAttributeRecPipeline
from .. import utils as serving_utils
from ..app import AppConfig, create_app
from ..models import Response, ResultResponse


class InferRequest(BaseModel):
    image: str


BoundingBox: TypeAlias = Annotated[List[float], Field(min_length=4, max_length=4)]


class Attribute(BaseModel):
    label: str
    score: float


class Vehicle(BaseModel):
    bbox: BoundingBox
    attributes: List[Attribute]
    score: float


class InferResult(BaseModel):
    vehicles: List[Vehicle]
    image: str


def create_pipeline_app(
    pipeline: VehicleAttributeRecPipeline, app_config: AppConfig
) -> FastAPI:
    app, ctx = create_app(
        pipeline=pipeline, app_config=app_config, app_aiohttp_session=True
    )

    @app.post(
        "/vehicle-attribute-recognition",
        operation_id="infer",
        responses={422: {"model": Response}},
    )
    async def _infer(request: InferRequest) -> ResultResponse[InferResult]:
        pipeline = ctx.pipeline
        aiohttp_session = ctx.aiohttp_session

        try:
            file_bytes = await serving_utils.get_raw_bytes(
                request.image, aiohttp_session
            )
            image = serving_utils.image_bytes_to_array(file_bytes)

            result = (await pipeline.infer(image))[0]

            vehicles: List[Vehicle] = []
            for obj in result["boxes"]:
                vehicles.append(
                    Vehicle(
                        bbox=obj["coordinate"],
                        attributes=[
                            Attribute(label=l, score=s)
                            for l, s in zip(obj["labels"], obj["cls_scores"])
                        ],
                        score=obj["det_score"],
                    )
                )
            output_image_base64 = serving_utils.image_to_base64(result.img)

            return ResultResponse(
                logId=serving_utils.generate_log_id(),
                errorCode=0,
                errorMsg="Success",
                result=InferResult(vehicles=vehicles, image=output_image_base64),
            )

        except Exception as e:
            logging.exception(e)
            raise HTTPException(status_code=500, detail="Internal server error")

    return app
