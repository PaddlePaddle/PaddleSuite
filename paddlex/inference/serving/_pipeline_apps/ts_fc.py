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

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ....utils import logging
from .. import utils as serving_utils
from ..app import AppConfig, create_app
from ..models import NoResultResponse, ResultResponse


class InferRequest(BaseModel):
    csv: str


class InferResult(BaseModel):
    csv: str


def create_pipeline_app(pipeline: Any, app_config: AppConfig) -> FastAPI:
    app, ctx = create_app(
        pipeline=pipeline, app_config=app_config, app_aiohttp_session=True
    )

    @app.post(
        "/time-series-forecasting",
        operation_id="infer",
        responses={422: {"model": NoResultResponse}},
        response_model_exclude_none=True,
    )
    async def _infer(request: InferRequest) -> ResultResponse[InferResult]:
        pipeline = ctx.pipeline
        aiohttp_session = ctx.aiohttp_session

        try:
            file_bytes = await serving_utils.get_raw_bytes(request.csv, aiohttp_session)
            df = serving_utils.csv_bytes_to_data_frame(file_bytes)

            result = (await pipeline.infer(df))[0]

            output_csv = serving_utils.base64_encode(
                serving_utils.data_frame_to_bytes(result["forecast"])
            )

            return ResultResponse[InferResult](
                logId=serving_utils.generate_log_id(),
                result=InferResult(csv=output_csv),
            )

        except Exception:
            logging.exception("Unexpected exception")
            raise HTTPException(status_code=500, detail="Internal server error")

    return app
