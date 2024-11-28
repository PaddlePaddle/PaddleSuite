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

from pathlib import Path
from typing import Any, Dict, Optional
from .base import BasePipeline
from ...utils.config import parse_config

# from .single_model_pipeline import (
#     _SingleModelPipeline,
#     ImageClassification,
#     ObjectDetection,
#     InstanceSegmentation,
#     SemanticSegmentation,
#     TSFc,
#     TSAd,
#     TSCls,
#     MultiLableImageClas,
#     SmallObjDet,
#     AnomalyDetection,
# )
# from .ocr import OCRPipeline
# from .formula_recognition import FormulaRecognitionPipeline
# from .table_recognition import TableRecPipeline
# from .face_recognition import FaceRecPipeline
# from .seal_recognition import SealOCRPipeline
# from .ppchatocrv3 import PPChatOCRPipeline
# from .layout_parsing import LayoutParsingPipeline
# from .pp_shitu_v2 import ShiTuV2Pipeline
# from .attribute_recognition import AttributeRecPipeline

from .ocr import OCRPipeline
from .doc_preprocessor import DocPreprocessorPipeline
from .layout_parsing import LayoutParsingPipeline
from .pp_chatocrv3_doc import PP_ChatOCRv3_doc_Pipeline

def get_pipeline_path(pipeline_name):
    pipeline_path = (
        Path(__file__).parent.parent.parent / "configs/pipelines" / f"{pipeline_name}.yaml"
    ).resolve()
    if not Path(pipeline_path).exists():
        return None
    return pipeline_path

def load_pipeline_config(pipeline_name: str) -> Dict[str, Any]:
    if not Path(pipeline_name).exists():
        pipeline_path = get_pipeline_path(pipeline_name)
        if pipeline_path is None:
            raise Exception(
                f"The pipeline ({pipeline_name}) does not exist! Please use a pipeline name or a config file path!"
            )
    else:
        pipeline_path = pipeline_name
    config = parse_config(pipeline_path)
    return config

def create_pipeline(
    pipeline: str,
    device=None,
    pp_option=None,
    use_hpip: bool = False,
    hpi_params: Optional[Dict[str, Any]] = None,
    *args,
    **kwargs,
) -> BasePipeline:
    """build model evaluater

    Args:
        pipeline (str): the pipeline name, that is name of pipeline class

    Returns:
        BasePipeline: the pipeline, which is subclass of BasePipeline.
    """
    pipeline_name = pipeline
    config = load_pipeline_config(pipeline_name)
    assert pipeline_name == config["pipeline_name"]
    pipeline = BasePipeline.get(pipeline_name)(
        config=config,
        device=device,
        pp_option=pp_option,
        use_hpip=use_hpip,
        hpi_params=hpi_params)
    return pipeline
    