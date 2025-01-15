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

from typing import Any, Dict, Optional
import numpy as np

from ...utils.pp_option import PaddlePredictorOption
from ..base import BasePipeline

# [TODO] 待更新models_new到models
from ...models_new.object_detection.result import DetResult


class ObjectDetectionPipeline(BasePipeline):
    """Object Detection Pipeline"""

    entities = "object_detection"

    def __init__(
        self,
        config: Dict,
        device: str = None,
        pp_option: PaddlePredictorOption = None,
        use_hpip: bool = False,
    ) -> None:
        """
        Initializes the class with given configurations and options.

        Args:
            config (Dict): Configuration dictionary containing model and other parameters.
            device (str): The device to run the prediction on. Default is None.
            pp_option (PaddlePredictorOption): Options for PaddlePaddle predictor. Default is None.
            use_hpip (bool): Whether to use high-performance inference (hpip) for prediction. Defaults to False.
        """
        super().__init__(device=device, pp_option=pp_option, use_hpip=use_hpip)
        model_cfg = config["SubModules"]["ObjectDetection"]
        model_kwargs = {}
        if "threshold" in model_cfg:
            model_kwargs["threshold"] = model_cfg["threshold"]
        if "img_size" in model_cfg:
            model_kwargs["img_size"] = model_cfg["img_size"]
        self.det_model = self.create_model(model_cfg, **model_kwargs)

    def predict(
        self,
        input: str | list[str] | np.ndarray | list[np.ndarray],
        threshold: Optional[float] = None,
        **kwargs,
    ) -> DetResult:
        """Predicts object detection results for the given input.

        Args:
            input (str | list[str] | np.ndarray | list[np.ndarray]): The input image(s) or path(s) to the images.
            threshold (Optional[float]): The threshold value to filter out low-confidence predictions. Default is None.
            **kwargs: Additional keyword arguments that can be passed to the function.
        Returns:
            DetResult: The predicted detection results.
        """
        yield from self.det_model(input, threshold=threshold, **kwargs)
