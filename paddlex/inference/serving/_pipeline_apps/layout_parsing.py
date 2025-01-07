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

from typing import Any, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing_extensions import Annotated, Literal, TypeAlias

from ....utils import logging
from .. import _utils as serving_utils
from .._app import AppConfig, create_app, main_operation
from .._models import DataInfo, ResultResponse
from ._common import cv as cv_common
from ._common import ocr as ocr_common


class InferRequest(ocr_common.InferRequest):
    useImgOrientationCls: bool = True
    useImgUnwarping: bool = True
    useSealTextDet: bool = True


BoundingBox: TypeAlias = Annotated[List[float], Field(min_length=4, max_length=4)]


class LayoutElement(BaseModel):
    bbox: BoundingBox
    label: str
    text: str
    layoutType: Literal["single", "double"]
    image: Optional[str] = None


class LayoutParsingResult(BaseModel):
    layoutElements: List[LayoutElement]


class InferResult(BaseModel):
    layoutParsingResults: List[LayoutParsingResult]
    dataInfo: DataInfo


def create_pipeline_app(pipeline: Any, app_config: AppConfig) -> FastAPI:
    app, ctx = create_app(
        pipeline=pipeline, app_config=app_config, app_aiohttp_session=True
    )

    ocr_common.update_app_context(ctx)

    @main_operation(
        app,
        "/layout-parsing",
        "infer",
    )
    async def _infer(
        request: InferRequest,
    ) -> ResultResponse[InferResult]:
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

        result = await pipeline.infer(
            images,
            use_doc_image_ori_cls_model=request.useImgOrientationCls,
            use_doc_image_unwarp_model=request.useImgUnwarping,
            use_seal_text_det_model=request.useSealTextDet,
        )

        layout_parsing_results: List[LayoutParsingResult] = []
        for i, item in enumerate(result):
            layout_elements: List[LayoutElement] = []
            for j, subitem in enumerate(
                item["layout_parsing_result"]["parsing_result"]
            ):
                dyn_keys = subitem.keys() - {"input_path", "layout_bbox", "layout"}
                if len(dyn_keys) != 1:
                    logging.error("Unexpected result: %s", subitem)
                    raise HTTPException(
                        status_code=500,
                        detail="Internal server error",
                    )
                label = next(iter(dyn_keys))
                if label in ("image", "figure", "img", "fig"):
                    text = subitem[label]["image_text"]
                    image = await serving_utils.call_async(
                        cv_common.postprocess_image,
                        subitem[label]["img"],
                        log_id=log_id,
                        filename=f"image_{i}_{j}.jpg",
                        file_storage=ctx.extra["file_storage"],
                        return_url=ctx.extra["return_img_urls"],
                        max_img_size=ctx.extra["max_output_img_size"],
                    )
                else:
                    text = subitem[label]
                    image = None
                layout_elements.append(
                    LayoutElement(
                        bbox=subitem["layout_bbox"],
                        label=label,
                        text=text,
                        layoutType=subitem["layout"],
                        image=image,
                    )
                )
            layout_parsing_results.append(
                LayoutParsingResult(layoutElements=layout_elements)
            )

        return ResultResponse[InferResult](
            logId=log_id,
            result=InferResult(
                layoutParsingResults=layout_parsing_results,
                dataInfo=data_info,
            ),
        )

    return app
