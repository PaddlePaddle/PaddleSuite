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
import tempfile
from typing import Any, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypeAlias

from .. import _utils as serving_utils
from .._app import AppConfig, create_app, main_operation
from .._models import DataInfo, ResultResponse
from ._common import ocr as ocr_common


class InferenceParams(BaseModel):
    maxLongSide: Optional[Annotated[int, Field(gt=0)]] = None


class AnalyzeImagesRequest(ocr_common.InferRequest):
    useGeneralOcr: bool = True
    useSealRecognition: bool = True
    useTableRecognition: bool = True
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


class VisualResult(BaseModel):
    texts: List[Text]
    tables: List[Table]
    inputImage: Optional[str] = None
    layoutImage: Optional[str] = None
    ocrImage: Optional[str] = None


class AnalyzeImagesResult(BaseModel):
    visualResults: List[VisualResult]
    visualInfo: dict
    dataInfo: DataInfo


class BuildVectorStoreRequest(BaseModel):
    visualInfo: dict
    minCharacters: int = 3500
    llmRequestInterval: float = 1.0


class BuildVectorStoreResult(BaseModel):
    vectorInfo: dict


class ChatRequest(BaseModel):
    keyList: List[str]
    visualInfo: dict
    useVectorRetrieval: bool = True
    vectorInfo: Optional[dict] = None
    minCharacters: int = 3500
    textTaskDescription: Optional[str] = None
    textOutputFormat: Optional[str] = None
    # Is the "Str" in the name unnecessary? Keep the names consistent with the
    # parameters of the wrapped function though.
    textRulesStr: Optional[str] = None
    # Should this be just "text" instead of "text content", given that there is
    # no container?
    textFewShotDemoTextContent: Optional[str] = None
    textFewShotDemoKeyValueList: Optional[str] = None
    tableTaskDescription: Optional[str] = None
    tableOutputFormat: Optional[str] = None
    tableRulesStr: Optional[str] = None
    tableFewShotDemoTextContent: Optional[str] = None
    tableFewShotDemoKeyValueList: Optional[str] = None


class ChatResult(BaseModel):
    chatResult: dict


# XXX: Since the pipeline class does not provide serialization and
# deserialization methods, these are implemented here based on the save-to-path
# and load-from-path methods.
def _serialize_vector_info(pipeline: Any, vector_info: dict) -> str:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        pipeline.save_vector(vector_info, path)
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    finally:
        os.unlink(path)


def _deserialize_vector_info(pipeline: Any, vector_info: str) -> dict:
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", suffix=".json", delete=False
    ) as f:
        f.write(vector_info)
        path = f.name
    try:
        return pipeline.load_vector(path)
    finally:
        os.unlink(path)


def create_pipeline_app(pipeline: Any, app_config: AppConfig) -> FastAPI:
    app, ctx = create_app(
        pipeline=pipeline, app_config=app_config, app_aiohttp_session=True
    )

    ocr_common.update_app_context(ctx)

    @main_operation(
        app,
        "/chatocr-visual",
        "analyzeImages",
    )
    async def _analyze_images(
        request: AnalyzeImagesRequest,
    ) -> ResultResponse[AnalyzeImagesResult]:
        pipeline = ctx.pipeline

        log_id = serving_utils.generate_log_id()

        if request.inferenceParams is not None:
            max_long_side = request.inferenceParams.maxLongSide
        else:
            max_long_side = None

        images, data_info = await ocr_common.get_images(request, ctx)

        result = await pipeline.call(
            pipeline.pipeline.visual_predict,
            images,
            use_doc_orientation_classify=request.useDocOrientationClassify,
            use_doc_unwarping=request.useDocUnwarping,
            use_general_ocr=request.useGeneralOcr,
            use_seal_recognition=request.useSealRecognition,
            use_table_recognition=request.useTableRecognition,
        )

        visual_results: List[VisualResult] = []
        for i, (img, item) in enumerate(zip(images, result["layout_parsing_result"])):
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
            visual_result = VisualResult(
                texts=texts,
                tables=tables,
                inputImage=input_img,
                layoutImage=layout_img,
                ocrImage=ocr_img,
            )
            visual_results.append(visual_result)

        return ResultResponse[AnalyzeImagesResult](
            logId=log_id,
            result=AnalyzeImagesResult(
                visualResults=visual_results,
                visualInfo=result["visual_info"],
                dataInfo=data_info,
            ),
        )

    @main_operation(
        app,
        "/chatocr-vector",
        "buildVectorStore",
    )
    async def _build_vector_store(
        request: BuildVectorStoreRequest,
    ) -> ResultResponse[BuildVectorStoreResult]:
        pipeline = ctx.pipeline

        vector_info = await serving_utils.call_async(
            pipeline.pipeline.build_vector,
            request.visualInfo,
            min_characters=request.minCharacters,
            llm_request_interval=request.llmRequestInterval,
        )

        vector_info = await serving_utils.call_async(
            _serialize_vector_info, pipeline.pipeline, vector_info
        )

        return ResultResponse[BuildVectorStoreResult](
            logId=serving_utils.generate_log_id(),
            result=BuildVectorStoreResult(vectorInfo=vector_info),
        )

    @main_operation(
        app,
        "/chatocr-chat",
        "chat",
    )
    async def _chat(
        request: ChatRequest,
    ) -> ResultResponse[ChatResult]:
        pipeline = ctx.pipeline

        if request.vectorInfo:
            vector_info = await serving_utils.call_async(
                _deserialize_vector_info,
                pipeline.pipeline,
                request.vectorInfo,
            )
        else:
            vector_info = None

        result = await serving_utils.call_async(
            pipeline.pipeline.chat,
            request.keyList,
            request.visualInfo,
            use_vector_retrieval=request.useVectorRetrieval,
            vector_info=vector_info,
            min_characters=request.minCharacters,
            text_task_description=request.textTaskDescription,
            text_output_format=request.textOutputFormat,
            text_rules_str=request.textRulesStr,
            text_few_shot_demo_text_content=request.textFewShotDemoTextContent,
            text_few_shot_demo_key_value_list=request.textFewShotDemoKeyValueList,
            table_task_description=request.tableTaskDescription,
            table_output_format=request.tableOutputFormat,
            table_rules_str=request.tableRulesStr,
            table_few_shot_demo_text_content=request.tableFewShotDemoTextContent,
            table_few_shot_demo_key_value_list=request.tableFewShotDemoKeyValueList,
        )

        chat_result = ChatResult(
            chatResult=result["chat_res"],
        )

        return ResultResponse[ChatResult](
            logId=serving_utils.generate_log_id(),
            result=chat_result,
        )

    return app
