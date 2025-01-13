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
        device: Optional[str] = None,
        use_doc_orientation_classify: Optional[bool] = None,
        use_doc_unwarping: Optional[bool] = None,
        use_textline_orientation: Optional[bool] = None,
        limit_side_len: Optional[int] = None,
        limit_type: Optional[str] = None,
        thresh: Optional[float] = None,
        box_thresh: Optional[float] = None,
        max_candidates: Optional[int] = None,
        unclip_ratio: Optional[float] = None,
        use_dilation: Optional[bool] = None,
        pp_option: Optional[PaddlePredictorOption] = None,
        use_hpip: bool = False,
        hpi_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initializes the class with given configurations and options.

        Args:
            config (Dict): Configuration dictionary containing model and other parameters.
            device (Union[str, None]): The device to run the prediction on.
            use_textline_orientation (Union[bool, None]): Whether to use textline orientation.
            use_doc_orientation_classify (Union[bool, None]): Whether to use document orientation classification.
            use_doc_unwarping (Union[bool, None]): Whether to use document unwarping.
            limit_side_len (Union[int, None]): Limit of side length.
            limit_type (Union[str, None]): Type of limit.
            thresh (Union[float, None]): Threshold value.
            box_thresh (Union[float, None]): Box threshold value.
            max_candidates (Union[int, None]): Maximum number of candidates.
            unclip_ratio (Union[float, None]): Unclip ratio.
            use_dilation (Union[bool, None]): Whether to use dilation.
            pp_option (Union[PaddlePredictorOption, None]): Options for PaddlePaddle predictor.
            use_hpip (Union[bool, None]): Whether to use high-performance inference.
            hpi_params (Union[Dict[str, Any], None]): HPIP specific parameters.
        """
        super().__init__(
            device=device, pp_option=pp_option, use_hpip=use_hpip, hpi_params=hpi_params
        )

        self.use_textline_orientation = (
            use_textline_orientation
            if use_textline_orientation is not None
            else config.get("use_textline_orientation", False)
        )
        self.use_doc_preprocessor = self.get_preprocessor_value(
            use_doc_orientation_classify, use_doc_unwarping, config, False
        )

        text_det_default_params = {
            "limit_side_len": 960,
            "limit_type": "max",
            "thresh": 0.3,
            "box_thresh": 0.6,
            "max_candidates": 1000,
            "unclip_ratio": 2.0,
            "use_dilation": False,
        }

        text_det_config = config["SubModules"]["TextDetection"]
        for key, default_params in text_det_default_params.items():
            text_det_config[key] = locals().get(
                key, text_det_config.get(key, default_params)
            )
        self.text_det_model = self.create_model(text_det_config)

        text_rec_config = config["SubModules"]["TextRecognition"]
        self.text_rec_model = self.create_model(text_rec_config)

        if self.use_textline_orientation:
            textline_orientation_config = config["SubModules"]["TextLineOrientation"]
            self.textline_orientation_model = self.create_model(
                textline_orientation_config
            )

        if self.use_doc_preprocessor:
            doc_preprocessor_config = config["SubPipelines"]["DocPreprocessor"]
            self.doc_preprocessor_pipeline = self.create_pipeline(
                doc_preprocessor_config
            )

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

    @staticmethod
    def get_preprocessor_value(orientation, unwarping, config, default):
        if orientation is None and unwarping is None:
            return config.get("use_doc_preprocessor", default)
        else:
            if orientation is False and unwarping is False:
                return False
            else:
                return True

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

    def check_model_settings_valid(self, model_settings: Dict) -> bool:
        """
        Check if the input parameters are valid based on the initialized models.

        Args:
            model_info_params(Dict): A dictionary containing input parameters.

        Returns:
            bool: True if all required models are initialized according to input parameters, False otherwise.
        """

        if model_settings["use_doc_preprocessor"] and not self.use_doc_preprocessor:
            logging.error(
                "Set use_doc_preprocessor, but the models for doc preprocessor are not initialized."
            )
            return False

        if (
            model_settings["use_textline_orientation"]
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
        else:
            doc_preprocessor_res = {"output_img": image_array}
        return doc_preprocessor_res

    def predict(
        self,
        input: str | list[str] | np.ndarray | list[np.ndarray],
        use_doc_orientation_classify: bool = False,
        use_doc_unwarping: bool = False,
        use_textline_orientation: bool = False,
        limit_side_len: int = 960,
        limit_type: str = "max",
        thresh: float = 0.3,
        box_thresh: float = 0.6,
        max_candidates: int = 1000,
        unclip_ratio: float = 2.0,
        use_dilation: bool = False,
        **kwargs,
    ) -> OCRResult:
        """Predicts OCR results for the given input.

        Args:
            input (str | list[str] | np.ndarray | list[np.ndarray]): The input image(s) or path(s) to the images or pdf(s).
            **kwargs: Additional keyword arguments that can be passed to the function.

        Returns:
            OCRResult: An iterable of OCRResult objects, each containing the predicted text and other relevant information.
        """

        model_settings = {
            "use_doc_orientation_classify": use_doc_orientation_classify,
            "use_doc_unwarping": use_doc_unwarping,
            "use_textline_orientation": use_textline_orientation,
        }
        if use_doc_orientation_classify or use_doc_unwarping:
            model_settings["use_doc_preprocessor"] = True
        else:
            model_settings["use_doc_preprocessor"] = False

        if not self.check_model_settings_valid(model_settings):
            yield None

        text_det_params = {
            "limit_side_len": limit_side_len,
            "limit_type": limit_type,
            "thresh": thresh,
            "box_thresh": box_thresh,
            "max_candidates": max_candidates,
            "unclip_ratio": unclip_ratio,
            "use_dilation": use_dilation,
        }

        for img_id, batch_data in enumerate(self.batch_sampler(input)):
            image_array = self.img_reader(batch_data)[0]
            img_id += 1

            doc_preprocessor_res = self.predict_doc_preprocessor_res(
                image_array, model_settings
            )
            doc_preprocessor_image = doc_preprocessor_res["output_img"]

            det_res = next(
                self.text_det_model(doc_preprocessor_image, **text_det_params)
            )

            dt_polys = det_res["dt_polys"]
            dt_scores = det_res["dt_scores"]

            dt_polys = self._sort_boxes(dt_polys)

            single_img_res = {
                "input_path": input,
                # TODO: `doc_preprocessor_image` parameter does not need to be retained here, it requires further confirmation.
                "doc_preprocessor_image": doc_preprocessor_image,
                "doc_preprocessor_res": doc_preprocessor_res,
                "dt_polys": dt_polys,
                "img_id": img_id,
                "input_params": model_settings,
                "text_det_params": text_det_params,
                "text_type": self.text_type,
            }

            single_img_res["rec_text"] = []
            single_img_res["rec_score"] = []
            if len(dt_polys) > 0:
                all_subs_of_img = list(
                    self._crop_by_polys(doc_preprocessor_image, dt_polys)
                )
                # use textline orientation model
                if model_settings["use_textline_orientation"]:
                    angles = [
                        textline_angle_info["class_ids"][0]
                        for textline_angle_info in self.textline_orientation_model(
                            all_subs_of_img
                        )
                    ]
                    single_img_res["textline_orientation_angle"] = angles
                    all_subs_of_img = self.rotate_image(all_subs_of_img, angles)

                for rec_res in self.text_rec_model(all_subs_of_img):
                    single_img_res["rec_text"].append(rec_res["rec_text"])
                    single_img_res["rec_score"].append(rec_res["rec_score"])

            yield OCRResult(single_img_res)
