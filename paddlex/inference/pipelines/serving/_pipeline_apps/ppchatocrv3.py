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

import asyncio
import os
from typing import Awaitable, Final, List, Literal, Optional, Tuple, Union

import numpy as np
from fastapi import FastAPI, HTTPException
from numpy.typing import ArrayLike
from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypeAlias, assert_never

from .....utils import logging
from .... import results
from ...ppchatocrv3 import PPChatOCRPipeline
from ..storage import SupportsGetURL, Storage, create_storage
from .. import utils as serving_utils
from ..app import AppConfig, create_app
from ..models import Response, ResultResponse

_DEFAULT_MAX_IMG_SIZE: Final[Tuple[int, int]] = (2000, 2000)
_DEFAULT_MAX_NUM_IMGS: Final[int] = 10


FileType: TypeAlias = Literal[0, 1]


class InferenceParams(BaseModel):
    maxLongSide: Optional[Annotated[int, Field(gt=0)]] = None


class AnalyzeImagesRequest(BaseModel):
    file: str
    fileType: Optional[FileType] = None
    useImgOrientationCls: bool = True
    useImgUnwrapping: bool = True
    useSealTextDet: bool = True
    inferenceParams: Optional[InferenceParams] = None


Point: TypeAlias = Annotated[List[int], Field(min_length=2, max_length=2)]
Polygon: TypeAlias = Annotated[List[Point], Field(min_length=3)]
BoundingBox: TypeAlias = Annotated[List[float], Field(min_length=4, max_length=4)]


class Text(BaseModel):
    poly: Polygon
    text: str
    score: float


class Table(BaseModel):
    bbox: BoundingBox
    html: str


class VisionResult(BaseModel):
    texts: List[Text]
    tables: List[Table]
    inputImage: str
    ocrImage: str
    layoutImage: str


class AnalyzeImagesResult(BaseModel):
    visionResults: List[VisionResult]
    visionInfo: dict


class QianfanParams(BaseModel):
    apiKey: str
    secretKey: str
    apiType: Literal["qianfan"] = "qianfan"


class AIStudioParams(BaseModel):
    accessToken: str
    apiType: Literal["aistudio"] = "aistudio"


LLMName: TypeAlias = Literal[
    "ernie-3.5",
    "ernie-3.5-8k",
    "ernie-lite",
    "ernie-4.0",
    "ernie-4.0-turbo-8k",
    "ernie-speed",
    "ernie-speed-128k",
    "ernie-tiny-8k",
    "ernie-char-8k",
]
LLMParams: TypeAlias = Union[QianfanParams, AIStudioParams]


class BuildVectorStoreRequest(BaseModel):
    visionInfo: dict
    minChars: Optional[int] = None
    llmRequestInterval: Optional[float] = None
    llmName: Optional[LLMName] = None
    llmParams: Optional[Annotated[LLMParams, Field(discriminator="apiType")]] = None


class BuildVectorStoreResult(BaseModel):
    vectorStore: str


class RetrieveKnowledgeRequest(BaseModel):
    keys: List[str]
    vectorStore: str
    llmName: Optional[LLMName] = None
    llmParams: Optional[Annotated[LLMParams, Field(discriminator="apiType")]] = None


class RetrieveKnowledgeResult(BaseModel):
    retrievalResult: str


class ChatRequest(BaseModel):
    keys: List[str]
    visionInfo: dict
    vectorStore: Optional[str] = None
    retrievalResult: Optional[str] = None
    taskDescription: Optional[str] = None
    rules: Optional[str] = None
    fewShot: Optional[str] = None
    llmName: Optional[LLMName] = None
    llmParams: Optional[Annotated[LLMParams, Field(discriminator="apiType")]] = None
    returnPrompts: bool = False


class Prompts(BaseModel):
    ocr: str
    table: Optional[str] = None
    html: Optional[str] = None


class ChatResult(BaseModel):
    chatResult: dict
    prompts: Optional[Prompts] = None


def _llm_params_to_dict(llm_params: LLMParams) -> dict:
    if llm_params.apiType == "qianfan":
        return {
            "api_type": "qianfan",
            "ak": llm_params.apiKey,
            "sk": llm_params.secretKey,
        }
    if llm_params.apiType == "aistudio":
        return {"api_type": "aistudio", "access_token": llm_params.accessToken}
    else:
        assert_never(llm_params.apiType)


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


def create_pipeline_app(pipeline: PPChatOCRPipeline, app_config: AppConfig) -> FastAPI:
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
        "/chatocr-vision",
        operation_id="analyzeImages",
        responses={422: {"model": Response}},
    )
    async def _analyze_images(
        request: AnalyzeImagesRequest,
    ) -> ResultResponse[AnalyzeImagesResult]:
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

            result = await pipeline.call(
                pipeline.pipeline.visual_predict,
                images,
                use_doc_image_ori_cls_model=request.useImgOrientationCls,
                use_doc_image_unwarp_model=request.useImgUnwrapping,
                use_seal_text_det_model=request.useSealTextDet,
            )

            vision_results: List[VisionResult] = []
            for i, (img, item) in enumerate(zip(images, result[0])):
                pp_img_futures: List[Awaitable] = []
                future = serving_utils.call_async(
                    _postprocess_image,
                    img,
                    request_id=request_id,
                    filename=f"input_image_{i}.jpg",
                    file_storage=ctx.extra["file_storage"],
                )
                pp_img_futures.append(future)
                future = serving_utils.call_async(
                    _postprocess_image,
                    item["ocr_result"].img,
                    request_id=request_id,
                    filename=f"ocr_image_{i}.jpg",
                    file_storage=ctx.extra["file_storage"],
                )
                pp_img_futures.append(future)
                future = serving_utils.call_async(
                    _postprocess_image,
                    item["layout_result"].img,
                    request_id=request_id,
                    filename=f"layout_image_{i}.jpg",
                    file_storage=ctx.extra["file_storage"],
                )
                pp_img_futures.append(future)
                texts: List[Text] = []
                for poly, text, score in zip(
                    item["ocr_result"]["dt_polys"],
                    item["ocr_result"]["rec_text"],
                    item["ocr_result"]["rec_score"],
                ):
                    texts.append(Text(poly=poly, text=text, score=score))
                tables = [
                    Table(bbox=r["layout_bbox"], html=r["html"])
                    for r in item["table_result"]
                ]
                input_img, ocr_img, layout_img = await asyncio.gather(*pp_img_futures)
                vision_result = VisionResult(
                    texts=texts,
                    tables=tables,
                    inputImage=input_img,
                    ocrImage=ocr_img,
                    layoutImage=layout_img,
                )
                vision_results.append(vision_result)

            return ResultResponse(
                logId=serving_utils.generate_log_id(),
                errorCode=0,
                errorMsg="Success",
                result=AnalyzeImagesResult(
                    visionResults=vision_results,
                    visionInfo=result[1],
                ),
            )

        except Exception as e:
            logging.exception(e)
            raise HTTPException(status_code=500, detail="Internal server error")

    @app.post(
        "/chatocr-vector",
        operation_id="buildVectorStore",
        responses={422: {"model": Response}},
    )
    async def _build_vector_store(
        request: BuildVectorStoreRequest,
    ) -> ResultResponse[BuildVectorStoreResult]:
        pipeline = ctx.pipeline

        try:
            kwargs = {"visual_info": results.VisualInfoResult(request.visionInfo)}
            if request.minChars is not None:
                kwargs["min_characters"] = request.minChars
            if request.llmRequestInterval is not None:
                kwargs["llm_request_interval"] = request.llmRequestInterval
            if request.llmName is not None:
                kwargs["llm_name"] = request.llmName
            if request.llmParams is not None:
                kwargs["llm_params"] = _llm_params_to_dict(request.llmParams)

            result = await serving_utils.call_async(
                pipeline.pipeline.build_vector, **kwargs
            )

            return ResultResponse(
                logId=serving_utils.generate_log_id(),
                errorCode=0,
                errorMsg="Success",
                result=BuildVectorStoreResult(vectorStore=result["vector"]),
            )

        except Exception as e:
            logging.exception(e)
            raise HTTPException(status_code=500, detail="Internal server error")

    @app.post(
        "/chatocr-retrieval",
        operation_id="retrieveKnowledge",
        responses={422: {"model": Response}},
    )
    async def _retrieve_knowledge(
        request: RetrieveKnowledgeRequest,
    ) -> ResultResponse[RetrieveKnowledgeResult]:
        pipeline = ctx.pipeline

        try:
            kwargs = {
                "key_list": request.keys,
                "vector": results.VectorResult({"vector": request.vectorStore}),
            }
            if request.llmName is not None:
                kwargs["llm_name"] = request.llmName
            if request.llmParams is not None:
                kwargs["llm_params"] = _llm_params_to_dict(request.llmParams)

            result = await serving_utils.call_async(
                pipeline.pipeline.retrieval, **kwargs
            )

            return ResultResponse(
                logId=serving_utils.generate_log_id(),
                errorCode=0,
                errorMsg="Success",
                result=RetrieveKnowledgeResult(retrievalResult=result["retrieval"]),
            )

        except Exception as e:
            logging.exception(e)
            raise HTTPException(status_code=500, detail="Internal server error")

    @app.post(
        "/chatocr-chat",
        operation_id="chat",
        responses={422: {"model": Response}},
        response_model_exclude_none=True,
    )
    async def _chat(
        request: ChatRequest,
    ) -> ResultResponse[ChatResult]:
        pipeline = ctx.pipeline

        try:
            kwargs = {
                "key_list": request.keys,
                "visual_info": results.VisualInfoResult(request.visionInfo),
            }
            if request.vectorStore is not None:
                kwargs["vector"] = results.VectorResult({"vector": request.vectorStore})
            if request.retrievalResult is not None:
                kwargs["retrieval_result"] = results.RetrievalResult(
                    {"retrieval": request.retrievalResult}
                )
            if request.taskDescription is not None:
                kwargs["user_task_description"] = request.taskDescription
            if request.rules is not None:
                kwargs["rules"] = request.rules
            if request.fewShot is not None:
                kwargs["few_shot"] = request.fewShot
            if request.llmName is not None:
                kwargs["llm_name"] = request.llmName
            if request.llmParams is not None:
                kwargs["llm_params"] = _llm_params_to_dict(request.llmParams)
            kwargs["save_prompt"] = request.returnPrompts

            result = await serving_utils.call_async(pipeline.pipeline.chat, **kwargs)

            if result["prompt"]:
                prompts = Prompts(
                    ocr=result["prompt"]["ocr_prompt"],
                    table=result["prompt"]["table_prompt"] or None,
                    html=result["prompt"]["html_prompt"] or None,
                )
            else:
                prompts = None
            chat_result = ChatResult(
                chatResult=result["chat_res"],
                prompts=prompts,
            )

            return ResultResponse(
                logId=serving_utils.generate_log_id(),
                errorCode=0,
                errorMsg="Success",
                result=chat_result,
            )

        except Exception as e:
            logging.exception(e)
            raise HTTPException(status_code=500, detail="Internal server error")

    return app
