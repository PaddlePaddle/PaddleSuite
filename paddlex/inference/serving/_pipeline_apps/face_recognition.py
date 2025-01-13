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
from typing import Any, Dict, Final, List

from fastapi import FastAPI

from .. import _utils as serving_utils
from .._app import AppConfig, create_app, main_operation
from .._models import ResultResponse
from .._storage import create_storage
from ..schemas import face_recognition as schema
from ._common import image_recognition as ir_common

# XXX: Currently the implementations of the face recognition and PP-ShiTuV2
# pipeline apps overlap significantly. We should aim to facilitate code reuse,
# but is it acceptable to assume a strong similarity between these two
# pipelines?


DEFAULT_INDEX_DIR: Final[str] = ".index"


def create_pipeline_app(pipeline: Any, app_config: AppConfig) -> FastAPI:
    app, ctx = create_app(
        pipeline=pipeline, app_config=app_config, app_aiohttp_session=True
    )

    if ctx.config.extra and "index_storage" in ctx.config.extra:
        ctx.extra["index_storage"] = create_storage(ctx.config.extra["index_storage"])
    else:
        ctx.extra["index_storage"] = create_storage(
            {"type": "file_system", "directory": DEFAULT_INDEX_DIR}
        )

    @main_operation(
        app,
        schema.BUILD_INDEX_ENDPOINT,
        "buildIndex",
    )
    async def _build_index(
        request: schema.BuildIndexRequest,
    ) -> ResultResponse[schema.BuildIndexResult]:
        pipeline = ctx.pipeline
        aiohttp_session = ctx.aiohttp_session

        images = [pair.image for pair in request.imageLabelPairs]
        file_bytes_list = await asyncio.gather(
            *(serving_utils.get_raw_bytes(img, aiohttp_session) for img in images)
        )
        images = [serving_utils.image_bytes_to_array(item) for item in file_bytes_list]
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
        index_key = ir_common.generate_index_key()
        index_data_bytes = await serving_utils.call_async(
            ir_common.serialize_index_data, index_data
        )
        await serving_utils.call_async(index_storage.set, index_key, index_data_bytes)

        return ResultResponse[schema.BuildIndexResult](
            logId=serving_utils.generate_log_id(),
            result=schema.BuildIndexResult(indexKey=index_key, idMap=index_data.id_map),
        )

    @main_operation(
        app,
        schema.ADD_IMAGES_TO_INDEX_ENDPOINT,
        "addImagesToIndex",
    )
    async def _add_images_to_index(
        request: schema.AddImagesToIndexRequest,
    ) -> ResultResponse[schema.AddImagesToIndexResult]:
        pipeline = ctx.pipeline
        aiohttp_session = ctx.aiohttp_session

        images = [pair.image for pair in request.imageLabelPairs]
        file_bytes_list = await asyncio.gather(
            *(serving_utils.get_raw_bytes(img, aiohttp_session) for img in images)
        )
        images = [serving_utils.image_bytes_to_array(item) for item in file_bytes_list]
        labels = [pair.label for pair in request.imageLabelPairs]

        if request.indexKey is not None:
            index_storage = ctx.extra["index_storage"]
            index_data_bytes = await serving_utils.call_async(
                index_storage.get, request.indexKey
            )
            index_data = await serving_utils.call_async(
                ir_common.deserialize_index_data, index_data_bytes
            )
        else:
            index_data = None

        index_data = await pipeline.call(
            pipeline.pipeline.append_index, images, labels, index_data
        )

        index_data_bytes = await serving_utils.call_async(
            ir_common.serialize_index_data, index_data
        )
        await serving_utils.call_async(
            index_storage.set, request.indexKey, index_data_bytes
        )

        return ResultResponse[schema.AddImagesToIndexResult](
            logId=serving_utils.generate_log_id(),
            result=schema.AddImagesToIndexResult(idMap=index_data.id_map),
        )

    @main_operation(
        app,
        schema.REMOVE_IMAGES_FROM_INDEX_ENDPOINT,
        "removeImagesFromIndex",
    )
    async def _remove_images_from_index(
        request: schema.RemoveImagesFromIndexRequest,
    ) -> ResultResponse[schema.RemoveImagesFromIndexResult]:
        pipeline = ctx.pipeline

        if request.indexKey is not None:
            index_storage = ctx.extra["index_storage"]
            index_data_bytes = await serving_utils.call_async(
                index_storage.get, request.indexKey
            )
            index_data = await serving_utils.call_async(
                ir_common.deserialize_index_data, index_data_bytes
            )
        else:
            index_data = None

        index_data = await pipeline.call(
            pipeline.pipeline.remove_index, request.ids, index_data
        )

        index_data_bytes = await serving_utils.call_async(
            ir_common.serialize_index_data, index_data
        )
        await serving_utils.call_async(
            index_storage.set, request.indexKey, index_data_bytes
        )

        return ResultResponse[schema.RemoveImagesFromIndexResult](
            logId=serving_utils.generate_log_id(),
            result=schema.RemoveImagesFromIndexResult(idMap=index_data.id_map),
        )

    @main_operation(
        app,
        schema.INFER_ENDPOINT,
        "infer",
    )
    async def _infer(
        request: schema.InferRequest,
    ) -> ResultResponse[schema.InferResult]:
        pipeline = ctx.pipeline
        aiohttp_session = ctx.aiohttp_session

        image_bytes = await serving_utils.get_raw_bytes(request.image, aiohttp_session)
        image = serving_utils.image_bytes_to_array(image_bytes)

        if request.indexKey is not None:
            index_storage = ctx.extra["index_storage"]
            index_data_bytes = await serving_utils.call_async(
                index_storage.get, request.indexKey
            )
            index_data = await serving_utils.call_async(
                ir_common.deserialize_index_data, index_data_bytes
            )
        else:
            index_data = None

        if request.inferenceParams is not None:
            det_threshold = request.inferenceParams.detThreshold
            rec_threshold = request.inferenceParams.recThreshold
            top_k = request.inferenceParams.topK
        else:
            det_threshold = None
            rec_threshold = None
            top_k = None

        result = list(
            await pipeline.call(
                pipeline.pipeline.predict,
                image,
                index=index_data,
                det_threshold=det_threshold,
                rec_threshold=rec_threshold,
                top_k=top_k,
            )
        )[0]

        objs: List[Dict[str, Any]] = []
        for obj in result["boxes"]:
            rec_results: List[Dict[str, Any]] = []
            if obj["rec_scores"] is not None:
                for label, score in zip(obj["labels"], obj["rec_scores"]):
                    rec_results.append(
                        dict(
                            label=label,
                            score=score,
                        )
                    )
            objs.append(
                dict(
                    bbox=obj["coordinate"],
                    recResults=rec_results,
                    score=obj["det_score"],
                )
            )
        if ctx.config.visualize:
            output_image_base64 = serving_utils.base64_encode(
                serving_utils.image_to_bytes(result.img["res"])
            )
        else:
            output_image_base64 = None

        return ResultResponse[schema.InferResult](
            logId=serving_utils.generate_log_id(),
            result=schema.InferResult(faces=objs, image=output_image_base64),
        )

    return app
