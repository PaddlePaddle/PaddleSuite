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
from scipy.ndimage import rotate
import numpy as np
from ..base import BasePipeline
from .result import DocPreprocessorResult
from ....utils import logging
from ...common.reader import ReadImage
from ...common.batch_sampler import ImageBatchSampler
from ...utils.pp_option import PaddlePredictorOption


class DocPreprocessorPipeline(BasePipeline):
    """Doc Preprocessor Pipeline"""

    entities = "doc_preprocessor"

    def __init__(
        self,
        config: Dict,
        device: str = None,
        pp_option: PaddlePredictorOption = None,
        use_hpip: bool = False,
        hpi_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initializes the doc preprocessor pipeline.

        Args:
            config (Dict): Configuration dictionary containing various settings.
            device (str, optional): Device to run the predictions on. Defaults to None.
            pp_option (PaddlePredictorOption, optional): PaddlePredictor options. Defaults to None.
            use_hpip (bool, optional): Whether to use high-performance inference (hpip) for prediction. Defaults to False.
            hpi_params (Optional[Dict[str, Any]], optional): HPIP parameters. Defaults to None.
        """

        super().__init__(
            device=device, pp_option=pp_option, use_hpip=use_hpip, hpi_params=hpi_params
        )

        self.use_doc_orientation_classify = True
        if "use_doc_orientation_classify" in config:
            self.use_doc_orientation_classify = config["use_doc_orientation_classify"]

        self.use_doc_unwarping = True
        if "use_doc_unwarping" in config:
            self.use_doc_unwarping = config["use_doc_unwarping"]

        if self.use_doc_orientation_classify:
            doc_ori_classify_config = config["SubModules"]["DocOrientationClassify"]
            self.doc_ori_classify_model = self.create_model(doc_ori_classify_config)

        if self.use_doc_unwarping:
            doc_unwarping_config = config["SubModules"]["DocUnwarping"]
            self.doc_unwarping_model = self.create_model(doc_unwarping_config)

        self.batch_sampler = ImageBatchSampler(batch_size=1)
        self.img_reader = ReadImage(format="BGR")

    def rotate_image(self, image_array: np.ndarray, rotate_angle: float) -> np.ndarray:
        """
        Rotate the given image array by the specified angle.

        Args:
            image_array (np.ndarray): The input image array to be rotated.
            rotate_angle (float): The angle in degrees by which to rotate the image.

        Returns:
            np.ndarray: The rotated image array.

        Raises:
            AssertionError: If rotate_angle is not in the range [0, 360).
        """
        assert (
            rotate_angle >= 0 and rotate_angle < 360
        ), "rotate_angle must in [0-360), but get {rotate_angle}."
        return rotate(image_array, rotate_angle, reshape=True)

    def check_input_params_valid(self, input_params: Dict) -> bool:
        """
        Check if the input parameters are valid based on the initialized models.

        Args:
            input_params (Dict): A dictionary containing input parameters.

        Returns:
            bool: True if all required models are initialized according to input parameters, False otherwise.
        """

        if (
            input_params["use_doc_orientation_classify"]
            and not self.use_doc_orientation_classify
        ):
            logging.error(
                "Set use_doc_orientation_classify, but the model for doc orientation classify is not initialized."
            )
            return False

        if input_params["use_doc_unwarping"] and not self.use_doc_unwarping:
            logging.error(
                "Set use_doc_unwarping, but the model for doc unwarping is not initialized."
            )
            return False

        return True

    def predict(
        self,
        input: str | list[str] | np.ndarray | list[np.ndarray],
        use_doc_orientation_classify: bool = True,
        use_doc_unwarping: bool = False,
        **kwargs
    ) -> DocPreprocessorResult:
        """
        Predict the preprocessing result for the input image or images.

        Args:
            input (str | list[str] | np.ndarray | list[np.ndarray]): The input image(s) or path(s) to the images or pdfs.
            use_doc_orientation_classify (bool): Whether to use document orientation classification.
            use_doc_unwarping (bool): Whether to use document unwarping.
            **kwargs: Additional keyword arguments.

        Returns:
            DocPreprocessorResult: A generator yielding preprocessing results.
        """

        input_params = {
            "use_doc_orientation_classify": use_doc_orientation_classify,
            "use_doc_unwarping": use_doc_unwarping,
        }

        if not self.check_input_params_valid(input_params):
            yield {"error": "input params invalid"}

        for img_id, batch_data in enumerate(self.batch_sampler(input)):
            image_array = self.img_reader(batch_data)[0]

            if input_params["use_doc_orientation_classify"]:
                pred = next(self.doc_ori_classify_model(image_array))
                angle = int(pred["label_names"][0])
                rot_img = self.rotate_image(image_array, angle)
            else:
                angle = -1
                rot_img = image_array

            if input_params["use_doc_unwarping"]:
                output_img = next(self.doc_unwarping_model(rot_img))["doctr_img"]
            else:
                output_img = rot_img

            img_id += 1
            single_img_res = {
                "input_image": image_array,
                "input_params": input_params,
                "angle": angle,
                "rot_img": rot_img,
                "output_img": output_img,
                "img_id": img_id,
            }
            yield DocPreprocessorResult(single_img_res)
