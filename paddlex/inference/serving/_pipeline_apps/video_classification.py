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

import os
from typing import Any, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing_extensions import Annotated

from .. import _utils as serving_utils
from .._app import AppConfig, create_app, main_operation
from .._models import ResultResponse


class InferenceParams(BaseModel):
    topK: Optional[Annotated[int, Field(gt=0)]] = None


class InferRequest(BaseModel):
    video: str
    inferenceParams: Optional[InferenceParams] = None


class Category(BaseModel):
    id: int
    name: str
    score: float


class InferResult(BaseModel):
    categories: List[Category]


def create_pipeline_app(pipeline: Any, app_config: AppConfig) -> FastAPI:
    app, ctx = create_app(
        pipeline=pipeline, app_config=app_config, app_aiohttp_session=True
    )

    @main_operation(
        app,
        "/video-classification",
        "infer",
    )
    async def _infer(request: InferRequest) -> ResultResponse[InferResult]:
        pipeline = ctx.pipeline
        aiohttp_session = ctx.aiohttp_session

        file_bytes = await serving_utils.get_raw_bytes(request.video, aiohttp_session)
        video_path = await serving_utils.call_async(
            serving_utils.write_to_temp_file,
            file_bytes,
            suffix=serving_utils.infer_file_ext(request.video),
        )

        if request.inferenceParams is not None:
            top_k = request.inferenceParams.topK
        else:
            top_k = None

        try:
            result = (await pipeline.infer(video_path, topk=top_k))[0]
        finally:
            await serving_utils.call_async(os.unlink, video_path)

        if "label_names" in result:
            cat_names = result["label_names"]
        else:
            cat_names = [str(id_) for id_ in result["class_ids"]]
        categories: List[Category] = []
        for id_, name, score in zip(result["class_ids"], cat_names, result["scores"]):
            categories.append(Category(id=id_, name=name, score=score))

        return ResultResponse[InferResult](
            logId=serving_utils.generate_log_id(),
            result=InferResult(categories=categories),
        )

    return app
