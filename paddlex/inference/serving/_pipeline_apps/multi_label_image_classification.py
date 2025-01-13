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

from typing import Any, Dict, List

from fastapi import FastAPI

from .. import _utils as serving_utils
from .._app import AppConfig, create_app, main_operation
from .._models import ResultResponse
from ..schemas.multi_label_image_classification import (
    INFER_ENDPOINT,
    InferRequest,
    InferResult,
)


def create_pipeline_app(pipeline: Any, app_config: AppConfig) -> FastAPI:
    app, ctx = create_app(
        pipeline=pipeline, app_config=app_config, app_aiohttp_session=True
    )

    @main_operation(
        app,
        INFER_ENDPOINT,
        "infer",
    )
    async def _infer(request: InferRequest) -> ResultResponse[InferResult]:
        pipeline = ctx.pipeline
        aiohttp_session = ctx.aiohttp_session

        if request.inferenceParams is not None:
            threshold = request.inferenceParams.threshold
        else:
            threshold = None

        file_bytes = await serving_utils.get_raw_bytes(request.image, aiohttp_session)
        image = serving_utils.image_bytes_to_array(file_bytes)

        result = (await pipeline.infer(image, threshold=threshold))[0]

        if "label_names" in result:
            cat_names = result["label_names"]
        else:
            cat_names = [str(id_) for id_ in result["class_ids"]]
        categories: List[Dict[str, Any]] = []
        for id_, name, score in zip(result["class_ids"], cat_names, result["scores"]):
            categories.append(dict(id=id_, name=name, score=score))
        if ctx.config.visualize:
            output_image_base64 = serving_utils.base64_encode(
                serving_utils.image_to_bytes(result.img["res"])
            )
        else:
            output_image_base64 = None

        return ResultResponse[InferResult](
            logId=serving_utils.generate_log_id(),
            result=InferResult(categories=categories, image=output_image_base64),
        )

    return app
