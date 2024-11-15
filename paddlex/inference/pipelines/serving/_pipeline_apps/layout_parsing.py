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
from typing import Final, List, Literal, Optional, Tuple

import numpy as np
from fastapi import FastAPI, HTTPException
from numpy.typing import ArrayLike
from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypeAlias

from .....utils import logging
from ...layout_parsing import LayoutParsingPipeline
from ..storage import SupportsGetURL, Storage, create_storage
from .. import utils as serving_utils
from ..app import AppConfig, create_app
from ..models import Response, ResultResponse

_DEFAULT_MAX_IMG_SIZE: Final[Tuple[int, int]] = (2000, 2000)
_DEFAULT_MAX_NUM_IMGS: Final[int] = 10


FileType: TypeAlias = Literal[0, 1]


class InferenceParams(BaseModel):
    maxLongSide: Optional[Annotated[int, Field(gt=0)]] = None


class InferRequest(BaseModel):
    file: str
    fileType: Optional[FileType] = None
    useImgOrientationCls: bool = True
    useImgUnwrapping: bool = True
    useSealTextDet: bool = True
    inferenceParams: Optional[InferenceParams] = None


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


def _postprocess_image(
    img: ArrayLike,
    request_id: str,
    filename: str,
    file_storage: Optional[Storage],
) -> str:
    key = f"{request_id}/{filename}"
    ext = os.path.splitext(filename)[1]
    img = np.asarray(img)
    img_bytes = serving_utils.image_array_to_bytes(img, ext=ext)
    if file_storage is not None:
        file_storage.set(key, img_bytes)
        if isinstance(file_storage, SupportsGetURL):
            return file_storage.get_url(key)
    return serving_utils.base64_encode(img_bytes)


def create_pipeline_app(
    pipeline: LayoutParsingPipeline, app_config: AppConfig
) -> FastAPI:
    app, ctx = create_app(
        pipeline=pipeline, app_config=app_config, app_aiohttp_session=True
    )

    if ctx.config.extra and "file_storage" in ctx.config.extra:
        ctx.extra["file_storage"] = create_storage(ctx.config.extra["file_storage"])
    else:
        ctx.extra["file_storage"] = None
    ctx.extra.setdefault("max_img_size", _DEFAULT_MAX_IMG_SIZE)
    ctx.extra.setdefault("max_num_imgs", _DEFAULT_MAX_NUM_IMGS)

    @app.post(
        "/layout-parsing",
        operation_id="infer",
        responses={422: {"model": Response}},
        response_model_exclude_none=True,
    )
    async def _infer(
        request: InferRequest,
    ) -> ResultResponse[InferResult]:
        pipeline = ctx.pipeline
        aiohttp_session = ctx.aiohttp_session

        request_id = serving_utils.generate_request_id()

        if request.fileType is None:
            if serving_utils.is_url(request.file):
                try:
                    file_type = serving_utils.infer_file_type(request.file)
                except Exception as e:
                    logging.exception(e)
                    raise HTTPException(
                        status_code=422,
                        detail="The file type cannot be inferred from the URL. Please specify the file type explicitly.",
                    )
            else:
                raise HTTPException(status_code=422, detail="Unknown file type")
        else:
            file_type = "PDF" if request.fileType == 0 else "IMAGE"

        if request.inferenceParams:
            max_long_side = request.inferenceParams.maxLongSide
            if max_long_side:
                raise HTTPException(
                    status_code=422,
                    detail="`max_long_side` is currently not supported.",
                )

        try:
            file_bytes = await serving_utils.get_raw_bytes(
                request.file, aiohttp_session
            )
            images = await serving_utils.call_async(
                serving_utils.file_to_images,
                file_bytes,
                file_type,
                max_img_size=ctx.extra["max_img_size"],
                max_num_imgs=ctx.extra["max_num_imgs"],
            )

            result = await pipeline.infer(
                images,
                use_doc_image_ori_cls_model=request.useImgOrientationCls,
                use_doc_image_unwarp_model=request.useImgUnwrapping,
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
                        raise RuntimeError(f"Unexpected result: {subitem}")
                    label = next(iter(dyn_keys))
                    if label in ("image", "figure", "img", "fig"):
                        image_ = await serving_utils.call_async(
                            _postprocess_image,
                            subitem[label]["img"],
                            request_id=request_id,
                            filename=f"image_{i}_{j}.jpg",
                            file_storage=ctx.extra["file_storage"],
                        )
                        text = subitem[label]["image_text"]
                    else:
                        image_ = None
                        text = subitem[label]
                    layout_elements.append(
                        LayoutElement(
                            bbox=subitem["layout_bbox"],
                            label=label,
                            text=text,
                            layoutType=subitem["layout"],
                            image=image_,
                        )
                    )
                layout_parsing_results.append(
                    LayoutParsingResult(layoutElements=layout_elements)
                )

            return ResultResponse(
                logId=serving_utils.generate_log_id(),
                errorCode=0,
                errorMsg="Success",
                result=InferResult(
                    layoutParsingResults=layout_parsing_results,
                ),
            )

        except Exception as e:
            logging.exception(e)
            raise HTTPException(status_code=500, detail="Internal server error")

    return app
