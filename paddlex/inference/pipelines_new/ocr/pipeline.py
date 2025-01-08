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

from typing import Any, Dict, List, Optional
import numpy as np
from scipy.ndimage import rotate
from ...common.reader import ReadImage
from ...common.batch_sampler import ImageBatchSampler
from ...utils.pp_option import PaddlePredictorOption
from ..base import BasePipeline
from ..components import CropByPolys, SortQuadBoxes, SortPolyBoxes
from .result import OCRResult
from ..doc_preprocessor.result import DocPreprocessorResult
from ....utils import logging


class OCRPipeline(BasePipeline):
    """OCR Pipeline"""

    entities = "OCR"

    def __init__(
        self,
        config: Dict,
        device: str = None,
        pp_option: PaddlePredictorOption = None,
        use_hpip: bool = False,
        hpi_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initializes the class with given configurations and options.

        Args:
            config (Dict): Configuration dictionary containing model and other parameters.
            device (str): The device to run the prediction on. Default is None.
            pp_option (PaddlePredictorOption): Options for PaddlePaddle predictor. Default is None.
            use_hpip (bool): Whether to use high-performance inference (hpip) for prediction. Defaults to False.
            hpi_params (Optional[Dict[str, Any]]): HPIP specific parameters. Default is None.
        """
        super().__init__(
            device=device, pp_option=pp_option, use_hpip=use_hpip, hpi_params=hpi_params
        )

        self.inintial_predictor(config)

        self.text_type = config["text_type"]

        if self.text_type == "general":
            self._sort_boxes = SortQuadBoxes()
            self._crop_by_polys = CropByPolys(det_box_type="quad")
        elif self.text_type == "seal":
            self._sort_boxes = SortPolyBoxes()
            self._crop_by_polys = CropByPolys(det_box_type="poly")
        else:
            raise ValueError("Unsupported text type {}".format(self.text_type))

        self.batch_sampler = ImageBatchSampler(batch_size=1)
        self.img_reader = ReadImage(format="BGR")

    def set_used_models_flag(self, config: Dict) -> None:
        """
        Set the flags for which models to use based on the configuration.

        Args:
            config (Dict): A dictionary containing configuration settings.

        Returns:
            None
        """
        pipeline_name = config["pipeline_name"]

        self.pipeline_name = pipeline_name

        self.use_doc_preprocessor = False

        if "use_doc_preprocessor" in config:
            self.use_doc_preprocessor = config["use_doc_preprocessor"]

        self.use_textline_orientation = False

        if "use_textline_orientation" in config:
            self.use_textline_orientation = config["use_textline_orientation"]

    def inintial_predictor(self, config: Dict) -> None:
        """Initializes the predictor based on the provided configuration.

        Args:
            config (Dict): A dictionary containing the configuration for the predictor.

        Returns:
            None
        """

        self.set_used_models_flag(config)

        text_det_model_config = config["SubModules"]["TextDetection"]
        self.text_det_model = self.create_model(text_det_model_config)

        text_rec_model_config = config["SubModules"]["TextRecognition"]
        self.text_rec_model = self.create_model(text_rec_model_config)

        if self.use_doc_preprocessor:
            doc_preprocessor_config = config["SubPipelines"]["DocPreprocessor"]
            self.doc_preprocessor_pipeline = self.create_pipeline(
                doc_preprocessor_config
            )
        # Just for initialize the predictor
        if self.use_textline_orientation:
            textline_orientation_config = config["SubModules"]["TextLineOrientation"]
            self.textline_orientation_model = self.create_model(
                textline_orientation_config
            )
        return

    def rotate_image(
        self, image_array_list: List[np.ndarray], rotate_angle_list: List[int]
    ) -> List[np.ndarray]:
        """
        Rotate the given image arrays by their corresponding angles.
        0 corresponds to 0 degrees, 1 corresponds to 180 degrees.

        Args:
            image_array_list (List[np.ndarray]): A list of input image arrays to be rotated.
            rotate_angle_list (List[int]): A list of rotation indicators (0 or 1).
                                        0 means rotate by 0 degrees
                                        1 means rotate by 180 degrees

        Returns:
            List[np.ndarray]: A list of rotated image arrays.

        Raises:
            AssertionError: If any rotate_angle is not 0 or 1.
            AssertionError: If the lengths of input lists don't match.
        """
        assert len(image_array_list) == len(
            rotate_angle_list
        ), f"Length of image_array_list ({len(image_array_list)}) must match length of rotate_angle_list ({len(rotate_angle_list)})"

        for angle in rotate_angle_list:
            assert angle in [0, 1], f"rotate_angle must be 0 or 1, now it's {angle}"

        rotated_images = []
        for image_array, rotate_indicator in zip(image_array_list, rotate_angle_list):
            # Convert 0/1 indicator to actual rotation angle
            rotate_angle = rotate_indicator * 180
            rotated_image = rotate(image_array, rotate_angle, reshape=True)
            rotated_images.append(rotated_image)

        return rotated_images

    def check_input_params_valid(self, input_params: Dict) -> bool:
        """
        Check if the input parameters are valid based on the initialized models.

        Args:
            input_params (Dict): A dictionary containing input parameters.

        Returns:
            bool: True if all required models are initialized according to input parameters, False otherwise.
        """

        if input_params["use_doc_preprocessor"] and not self.use_doc_preprocessor:
            logging.error(
                "Set use_doc_preprocessor, but the models for doc preprocessor are not initialized."
            )
            return False

        if (
            input_params["use_textline_orientation"]
            and not self.use_textline_orientation
        ):
            logging.error(
                "Set use_textline_orientation, but the models for use_textline_orientation are not initialized."
            )
            return False

        return True

    def predict_doc_preprocessor_res(
        self, image_array: np.ndarray, input_params: dict
    ) -> tuple[DocPreprocessorResult, np.ndarray]:
        """
        Preprocess the document image based on input parameters.

        Args:
            image_array (np.ndarray): The input image array.
            input_params (dict): Dictionary containing preprocessing parameters.

        Returns:
            tuple[DocPreprocessorResult, np.ndarray]: A tuple containing the preprocessing
                                              result dictionary and the processed image array.
        """
        if input_params["use_doc_preprocessor"]:
            use_doc_orientation_classify = input_params["use_doc_orientation_classify"]
            use_doc_unwarping = input_params["use_doc_unwarping"]
            doc_preprocessor_res = next(
                self.doc_preprocessor_pipeline(
                    image_array,
                    use_doc_orientation_classify=use_doc_orientation_classify,
                    use_doc_unwarping=use_doc_unwarping,
                )
            )
            doc_preprocessor_image = doc_preprocessor_res["output_img"]
        else:
            doc_preprocessor_res = {}
            doc_preprocessor_image = image_array
        return doc_preprocessor_res, doc_preprocessor_image

    def predict(
        self,
        input: str | list[str] | np.ndarray | list[np.ndarray],
        use_doc_orientation_classify: bool = False,
        use_doc_unwarping: bool = False,
        use_textline_orientation: bool = False,
        **kwargs,
    ) -> OCRResult:
        """Predicts OCR results for the given input.

        Args:
            input (str | list[str] | np.ndarray | list[np.ndarray]): The input image(s) or path(s) to the images or pdf(s).
            **kwargs: Additional keyword arguments that can be passed to the function.

        Returns:
            OCRResult: An iterable of OCRResult objects, each containing the predicted text and other relevant information.
        """

        input_params = {
            "use_doc_preprocessor": self.use_doc_preprocessor,
            "use_doc_orientation_classify": use_doc_orientation_classify,
            "use_doc_unwarping": use_doc_unwarping,
            "use_textline_orientation": self.use_textline_orientation,
        }
        if use_doc_orientation_classify or use_doc_unwarping:
            input_params["use_doc_preprocessor"] = True
        else:
            input_params["use_doc_preprocessor"] = False

        if not self.check_input_params_valid(input_params):
            yield None

        for img_id, batch_data in enumerate(self.batch_sampler(input)):
            image_array = self.img_reader(batch_data)[0]
            img_id += 1

            doc_preprocessor_res, doc_preprocessor_image = (
                self.predict_doc_preprocessor_res(image_array, input_params)
            )

            det_res = next(self.text_det_model(doc_preprocessor_image))

            dt_polys = det_res["dt_polys"]
            dt_scores = det_res["dt_scores"]

            ########## [TODO] Need to confirm filtering thresholds for detection and recognition modules

            dt_polys = self._sort_boxes(dt_polys)

            single_img_res = {
                "input_img": image_array,
                "doc_preprocessor_image": doc_preprocessor_image,
                "doc_preprocessor_res": doc_preprocessor_res,
                "dt_polys": dt_polys,
                "img_id": img_id,
                "input_params": input_params,
                "text_type": self.text_type,
            }

            single_img_res["rec_text"] = []
            single_img_res["rec_score"] = []
            if len(dt_polys) > 0:
                all_subs_of_img = list(
                    self._crop_by_polys(doc_preprocessor_image, dt_polys)
                )
                # use textline orientation model
                if input_params["use_textline_orientation"]:
                    angles = [
                        textline_angle_info["class_ids"][0]
                        for textline_angle_info in self.textline_orientation_model(
                            all_subs_of_img
                        )
                    ]
                    all_subs_of_img = self.rotate_image(all_subs_of_img, angles)

                for rec_res in self.text_rec_model(all_subs_of_img):
                    single_img_res["rec_text"].append(rec_res["rec_text"])
                    single_img_res["rec_score"].append(rec_res["rec_score"])

            yield OCRResult(single_img_res)
