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

from ...infra import utils as serving_utils
from ...infra.config import AppConfig
from ...infra.models import ResultResponse
from ...schemas.table_recognition import INFER_ENDPOINT, InferRequest, InferResult
from .._app import create_app, primary_operation
from ._common import ocr as ocr_common


def create_pipeline_app(pipeline: Any, app_config: AppConfig) -> FastAPI:
    app, ctx = create_app(
        pipeline=pipeline, app_config=app_config, app_aiohttp_session=True
    )

    ocr_common.update_app_context(ctx)

    @primary_operation(
        app,
        INFER_ENDPOINT,
        "infer",
    )
    async def _infer(request: InferRequest) -> ResultResponse[InferResult]:
        pipeline = ctx.pipeline

        log_id = serving_utils.generate_log_id()

        images, data_info = await ocr_common.get_images(request, ctx)

        result = await pipeline.infer(
            images,
            use_doc_orientation_classify=request.useDocOrientationClassify,
            use_doc_unwarping=request.useDocUnwarping,
        )

        table_rec_results: List[Dict[str, Any]] = []
        for i, (img, item) in enumerate(zip(images, result)):
            tables: List[Dict[str, Any]] = []
            for subitem in item["table_result"]:
                tables.append(
                    dict(
                        bbox=subitem["layout_bbox"],
                        html=subitem["html"],
                    )
                )
            if ctx.config.visualize:
                input_img, layout_img, ocr_img = await ocr_common.postprocess_images(
                    log_id=log_id,
                    index=i,
                    app_context=ctx,
                    input_image=img,
                    layout_image=item["layout_result"].img,
                    ocr_image=item["ocr_result"].img,
                )
            else:
                input_img, layout_img, ocr_img = None, None, None
            table_rec_results.append(
                dict(
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
