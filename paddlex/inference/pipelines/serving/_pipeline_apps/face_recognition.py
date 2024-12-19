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
import faiss
import pickle
import uuid
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypeAlias

from .....utils import logging
from ....components.retrieval.faiss import IndexData
from ...face_recognition import FaceRecPipeline
from ..storage import create_storage
from .. import utils as serving_utils
from ..app import AppConfig, create_app
from ..models import NoResultResponse, ResultResponse


class ImageLabelPair(BaseModel):
    image: str
    label: str


class BuildIndexRequest(BaseModel):
    imageLabelPairs: List[ImageLabelPair]


class BuildIndexResult(BaseModel):
    indexKey: str
    idMap: Dict[int, str]


class AddImagesToIndexRequest(BaseModel):
    imageLabelPairs: List[ImageLabelPair]
    indexKey: str


class AddImagesToIndexResult(BaseModel):
    idMap: Dict[int, str]


class RemoveImagesFromIndexRequest(BaseModel):
    ids: List[int]
    indexKey: str


class RemoveImagesFromIndexResult(BaseModel):
    idMap: Dict[int, str]


class InferRequest(BaseModel):
    image: str
    indexKey: Optional[str] = None


BoundingBox: TypeAlias = Annotated[List[float], Field(min_length=4, max_length=4)]


class RecResult(BaseModel):
    label: str
    score: float


class Face(BaseModel):
    bbox: BoundingBox
    recResults: List[RecResult]
    score: float


class InferResult(BaseModel):
    faces: List[Face]
    image: str


def _serialize_index_data(index_data: IndexData) -> bytes:
    tup = (index_data.index_bytes, index_data.index_info)
    return pickle.dumps(tup)


def _deserialize_index_data(index_data_bytes: bytes) -> IndexData:
    tup = pickle.loads(index_data_bytes)
    index = faiss.deserialize_index(tup[0])
    return IndexData(index, tup[1])


def create_pipeline_app(pipeline: FaceRecPipeline, app_config: AppConfig) -> FastAPI:
    app, ctx = create_app(
        pipeline=pipeline, app_config=app_config, app_aiohttp_session=True
    )

    if ctx.config.extra and "index_storage" in ctx.config.extra:
        ctx.extra["index_storage"] = create_storage(ctx.config.extra["index_storage"])
    else:
        ctx.extra["index_storage"] = create_storage({"type": "memory"})

    @app.post(
        "/face-recognition-index-build",
        operation_id="buildIndex",
        responses={422: {"model": NoResultResponse}},
        response_model_exclude_none=True,
    )
    async def _build_index(
        request: BuildIndexRequest,
    ) -> ResultResponse[BuildIndexResult]:
        pipeline = ctx.pipeline
        aiohttp_session = ctx.aiohttp_session

        try:
            images = [pair.image for pair in request.imageLabelPairs]
            file_bytes_list = await asyncio.gather(
                *(serving_utils.get_raw_bytes(img, aiohttp_session) for img in images)
            )
            images = [
                serving_utils.image_bytes_to_array(item) for item in file_bytes_list
            ]
            labels = [pair.label for pair in request.imageLabelPairs]

            # TODO: Support specifying `index_type` and `metric_type` in the
            # request
            index_data = await pipeline.call(
                pipeline.pipeline.build_index,
                images,
                labels,
                index_type="Flat",
                metric_type="IP",
            )

            index_storage = ctx.extra["index_storage"]
            index_key = str(uuid.uuid4())
            index_data_bytes = await serving_utils.call_async(
                _serialize_index_data, index_data
            )
            await serving_utils.call_async(
                index_storage.set, index_key, index_data_bytes
            )

            return ResultResponse[BuildIndexResult](
                logId=serving_utils.generate_log_id(),
                result=BuildIndexResult(indexKey=index_key, idMap=index_data.id_map),
            )

        except Exception:
            logging.exception("Unexpected exception")
            raise HTTPException(status_code=500, detail="Internal server error")

    @app.post(
        "/face-recognition-index-add",
        operation_id="buildIndex",
        responses={422: {"model": NoResultResponse}},
        response_model_exclude_none=True,
    )
    async def _add_images_to_index(
        request: AddImagesToIndexRequest,
    ) -> ResultResponse[AddImagesToIndexResult]:
        pipeline = ctx.pipeline
        aiohttp_session = ctx.aiohttp_session

        try:
            images = [pair.image for pair in request.imageLabelPairs]
            file_bytes_list = await asyncio.gather(
                *(serving_utils.get_raw_bytes(img, aiohttp_session) for img in images)
            )
            images = [
                serving_utils.image_bytes_to_array(item) for item in file_bytes_list
            ]
            labels = [pair.label for pair in request.imageLabelPairs]
            index_storage = ctx.extra["index_storage"]
            index_data_bytes = await serving_utils.call_async(
                index_storage.get, request.indexKey
            )
            index_data = await serving_utils.call_async(
                _deserialize_index_data, index_data_bytes
            )

            index_data = await pipeline.call(
                pipeline.pipeline.append_index, images, labels, index_data
            )

            index_data_bytes = await serving_utils.call_async(
                _serialize_index_data, index_data
            )
            await serving_utils.call_async(
                index_storage.set, request.indexKey, index_data_bytes
            )

            return ResultResponse[AddImagesToIndexResult](
                logId=serving_utils.generate_log_id(),
                result=AddImagesToIndexResult(idMap=index_data.id_map),
            )

        except Exception:
            logging.exception("Unexpected exception")
            raise HTTPException(status_code=500, detail="Internal server error")

    @app.post(
        "/face-recognition-index-remove",
        operation_id="buildIndex",
        responses={422: {"model": NoResultResponse}},
        response_model_exclude_none=True,
    )
    async def _remove_images_from_index(
        request: RemoveImagesFromIndexRequest,
    ) -> ResultResponse[RemoveImagesFromIndexResult]:
        pipeline = ctx.pipeline

        try:
            index_storage = ctx.extra["index_storage"]
            index_data_bytes = await serving_utils.call_async(
                index_storage.get, request.indexKey
            )
            index_data = await serving_utils.call_async(
                _deserialize_index_data, index_data_bytes
            )

            index_data = await pipeline.call(
                pipeline.pipeline.remove_index, request.ids, index_data
            )

            index_data_bytes = await serving_utils.call_async(
                _serialize_index_data, index_data
            )
            await serving_utils.call_async(
                index_storage.set, request.indexKey, index_data_bytes
            )

            return ResultResponse[RemoveImagesFromIndexResult](
                logId=serving_utils.generate_log_id(),
                result=RemoveImagesFromIndexResult(idMap=index_data.id_map),
            )

        except Exception:
            logging.exception("Unexpected exception")
            raise HTTPException(status_code=500, detail="Internal server error")

    @app.post(
        "/face-recognition-infer",
        operation_id="infer",
        responses={422: {"model": NoResultResponse}},
        response_model_exclude_none=True,
    )
    async def _infer(request: InferRequest) -> ResultResponse[InferResult]:
        pipeline = ctx.pipeline
        aiohttp_session = ctx.aiohttp_session

        try:
            image_bytes = await serving_utils.get_raw_bytes(
                request.image, aiohttp_session
            )
            image = serving_utils.image_bytes_to_array(image_bytes)

            if request.indexKey is not None:
                index_storage = ctx.extra["index_storage"]
                index_data_bytes = await serving_utils.call_async(
                    index_storage.get, request.indexKey
                )
                index_data = await serving_utils.call_async(
                    _deserialize_index_data, index_data_bytes
                )
            else:
                index_data = None

            result = list(
                await pipeline.call(pipeline.pipeline.predict, image, index=index_data)
            )[0]

            faces: List[Face] = []
            for face in result["boxes"]:
                rec_results: List[RecResult] = []
                if face["rec_scores"] is not None:
                    for label, score in zip(face["labels"], face["rec_scores"]):
                        rec_results.append(
                            RecResult(
                                label=label,
                                score=score,
                            )
                        )
                faces.append(
                    Face(
                        bbox=face["coordinate"],
                        recResults=rec_results,
                        score=face["det_score"],
                    )
                )
            output_image_base64 = serving_utils.base64_encode(
                serving_utils.image_to_bytes(result.img)
            )

            return ResultResponse[InferResult](
                logId=serving_utils.generate_log_id(),
                result=InferResult(faces=faces, image=output_image_base64),
            )

        except Exception:
            logging.exception("Unexpected exception")
            raise HTTPException(status_code=500, detail="Internal server error")

    return app
