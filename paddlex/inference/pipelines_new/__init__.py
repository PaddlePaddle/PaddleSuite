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
from ..utils.pp_option import PaddlePredictorOption
from .components import BaseChat, BaseRetriever, BaseGeneratePrompt
from ...utils.config import parse_config
from .ocr import OCRPipeline
from .doc_preprocessor import DocPreprocessorPipeline
from .layout_parsing import LayoutParsingPipeline
from .pp_chatocr import PP_ChatOCRv3_Pipeline, PP_ChatOCRv4_Pipeline
from .image_classification import ImageClassificationPipeline
from .seal_recognition import SealRecognitionPipeline
from .table_recognition import TableRecognitionPipeline
from .video_classification import VideoClassificationPipeline


def get_pipeline_path(pipeline_name: str) -> str:
    """
    Get the full path of the pipeline configuration file based on the provided pipeline name.

    Args:
        pipeline_name (str): The name of the pipeline.

    Returns:
        str: The full path to the pipeline configuration file or None if not found.
    """
    pipeline_path = (
        Path(__file__).parent.parent.parent
        / "configs/pipelines"
        / f"{pipeline_name}.yaml"
    ).resolve()
    if not Path(pipeline_path).exists():
        return None
    return pipeline_path


def load_pipeline_config(pipeline_name: str) -> Dict[str, Any]:
    """
    Load the pipeline configuration.

    Args:
        pipeline_name (str): The name of the pipeline or the path to the config file.

    Returns:
        Dict[str, Any]: The parsed pipeline configuration.

    Raises:
        Exception: If the config file of pipeline does not exist.
    """
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
    config: Dict = None,
    device: str = None,
    pp_option: PaddlePredictorOption = None,
    use_hpip: bool = False,
    hpi_params: Optional[Dict[str, Any]] = None,
    *args,
    **kwargs,
) -> BasePipeline:
    """
    Create a pipeline instance based on the provided parameters.
    If the input parameter config is not provided,
    it is obtained from the default config corresponding to the pipeline name.

    Args:
        pipeline (str): The name of the pipeline to create.
        config (Dict, optional): The path to the pipeline configuration file. Defaults to None.
        device (str, optional): The device to run the pipeline on. Defaults to None.
        pp_option (PaddlePredictorOption, optional): The options for the PaddlePredictor. Defaults to None.
        use_hpip (bool, optional): Whether to use high-performance inference (hpip) for prediction. Defaults to False.
        hpi_params (Optional[Dict[str, Any]], optional): Additional parameters for hpip. Defaults to None.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        BasePipeline: The created pipeline instance.
    """

    if config is None:
        config = load_pipeline_config(pipeline)
        pipeline_name = config["pipeline_name"]
    else:
        pipeline_name = pipeline

    pipeline = BasePipeline.get(pipeline_name)(
        config=config,
        device=device,
        pp_option=pp_option,
        use_hpip=use_hpip,
        hpi_params=hpi_params,
        *args,
        **kwargs,
    )
    return pipeline


def create_chat_bot(config: Dict, *args, **kwargs) -> BaseChat:
    """Creates an instance of a chat bot based on the provided configuration.

    Args:
        config (Dict): Configuration settings, expected to be a dictionary with at least a 'model_name' key.
        *args: Additional positional arguments. Not used in this function but allowed for future compatibility.
        **kwargs: Additional keyword arguments. Not used in this function but allowed for future compatibility.

    Returns:
        BaseChat: An instance of the chat bot class corresponding to the 'model_name' in the config.
    """
    model_name = config["model_name"]
    chat_bot = BaseChat.get(model_name)(config)
    return chat_bot


def create_retriever(
    config: Dict,
    *args,
    **kwargs,
) -> BaseRetriever:
    """
    Creates a retriever instance based on the provided configuration.

    Args:
        config (Dict): Configuration settings, expected to be a dictionary with at least a 'model_name' key.
        *args: Additional positional arguments. Not used in this function but allowed for future compatibility.
        **kwargs: Additional keyword arguments. Not used in this function but allowed for future compatibility.

    Returns:
        BaseRetriever: An instance of a retriever class corresponding to the 'model_name' in the config.
    """
    model_name = config["model_name"]
    retriever = BaseRetriever.get(model_name)(config)
    return retriever


def create_prompt_engeering(
    config: Dict,
    *args,
    **kwargs,
) -> BaseGeneratePrompt:
    """
    Creates a prompt engineering instance based on the provided configuration.

    Args:
        config (Dict): Configuration settings, expected to be a dictionary with at least a 'task_type' key.
        *args: Variable length argument list for additional positional arguments.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        BaseGeneratePrompt: An instance of a prompt engineering class corresponding to the 'task_type' in the config.
    """
    task_type = config["task_type"]
    pe = BaseGeneratePrompt.get(task_type)(config)
    return pe
