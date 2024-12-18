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

from typing import Final, List, Literal, Optional, Tuple

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypeAlias

from .....utils import logging
from ...formula_recognition import FormulaRecognitionPipeline
from .. import utils as serving_utils
from ..app import AppConfig, create_app
from ..models import NoResultResponse, ResultResponse

_DEFAULT_MAX_IMG_SIZE: Final[Tuple[int, int]] = (2000, 2000)
_DEFAULT_MAX_NUM_IMGS: Final[int] = 10

FileType: TypeAlias = Literal[0, 1]


class InferenceParams(BaseModel):
    maxLongSide: Optional[Annotated[int, Field(gt=0)]] = None


class InferRequest(BaseModel):
    file: str
    fileType: Optional[FileType] = None
    inferenceParams: Optional[InferenceParams] = None


Point: TypeAlias = Annotated[List[float], Field(min_length=2, max_length=2)]
Polygon: TypeAlias = Annotated[List[Point], Field(min_length=3)]


class Formula(BaseModel):
    poly: Polygon
    latex: str


class FormulaRecResult(BaseModel):
    formulas: List[Formula]
    layoutImage: str
    ocrImage: Optional[str] = None


class InferResult(BaseModel):
    formulaRecResults: List[FormulaRecResult]


def create_pipeline_app(
    pipeline: FormulaRecognitionPipeline, app_config: AppConfig
) -> FastAPI:
    app, ctx = create_app(
        pipeline=pipeline, app_config=app_config, app_aiohttp_session=True
    )

    ctx.extra["return_ocr_imgs"] = False
    ctx.extra["max_img_size"] = _DEFAULT_MAX_IMG_SIZE
    ctx.extra["max_num_imgs"] = _DEFAULT_MAX_NUM_IMGS
    if ctx.config.extra:
        if "return_ocr_imgs" in ctx.config.extra:
            ctx.extra["return_ocr_imgs"] = ctx.config.extra["return_ocr_imgs"]
        if "max_img_size" in ctx.config.extra:
            ctx.extra["max_img_size"] = ctx.config.extra["max_img_size"]
        if "max_num_imgs" in ctx.config.extra:
            ctx.extra["max_num_imgs"] = ctx.config.extra["max_num_imgs"]

    @app.post(
        "/formula-recognition",
        operation_id="infer",
        responses={422: {"model": NoResultResponse}},
        response_model_exclude_none=True,
    )
    async def _infer(request: InferRequest) -> ResultResponse[InferResult]:
        pipeline = ctx.pipeline
        aiohttp_session = ctx.aiohttp_session

        if request.fileType is None:
            if serving_utils.is_url(request.file):
                try:
                    file_type = serving_utils.infer_file_type(request.file)
                except Exception:
                    logging.exception("Failed to infer the file type")
                    raise HTTPException(
                        status_code=422,
                        detail="The file type cannot be inferred from the URL. Please specify the file type explicitly.",
                    )
            else:
                raise HTTPException(status_code=422, detail="Unknown file type")
        else:
            file_type = "PDF" if request.fileType == 0 else "IMAGE"

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

            result = await pipeline.infer(images)

            formula_rec_results: List[FormulaRecResult] = []
            for item in result:
                formulas: List[Formula] = []
                for poly, latex in zip(item["dt_polys"], item["rec_formula"]):
                    formulas.append(
                        Formula(
                            poly=poly,
                            latex=latex,
                        )
                    )
                layout_image_base64 = serving_utils.base64_encode(
                    serving_utils.image_to_bytes(item["layout_result"].img)
                )
                if ctx.extra["return_ocr_imgs"]:
                    ocr_image = item["formula_result"].img
                    ocr_image_base64 = serving_utils.base64_encode(
                        serving_utils.image_to_bytes(ocr_image)
                    )
                else:
                    ocr_image_base64 = None
                formula_rec_results.append(
                    FormulaRecResult(
                        formulas=formulas,
                        layoutImage=layout_image_base64,
                        ocrImage=ocr_image_base64,
                    )
                )

            return ResultResponse[InferResult](
                logId=serving_utils.generate_log_id(),
                result=InferResult(
                    formulaRecResults=formula_rec_results,
                ),
            )

        except Exception:
            logging.exception("Unexpected exception")
            raise HTTPException(status_code=500, detail="Internal server error")

    return app
