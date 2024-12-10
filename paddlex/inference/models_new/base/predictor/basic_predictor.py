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

from typing import Union, Tuple, List, Dict, Any, Iterator
from abc import abstractmethod

from .....utils.subclass_register import AutoRegisterABCMetaClass
from .....utils.flags import (
    INFER_BENCHMARK,
    INFER_BENCHMARK_WARMUP,
)
from .....utils import logging
from ....utils.pp_option import PaddlePredictorOption
from ....utils.benchmark import benchmark
from ....common.batch_sampler import BaseBatchSampler
from .base_predictor import BasePredictor


class PredictionWrap:
    """Wraps the prediction data and supports get by index."""

    def __init__(self, data: Dict[str, List[Any]], num: int) -> None:
        """Initializes the PredictionWrap with prediction data.

        Args:
            data (Dict[str, List[Any]]): A dictionary where keys are string identifiers and values are lists of predictions.
            num (int): The number of predictions, that is length of values per key in the data dictionary.

        Raises:
            AssertionError: If the length of any list in data does not match num.
        """
        assert isinstance(data, dict), "data must be a dictionary"
        for k in data:
            assert len(data[k]) == num, f"{len(data[k])} != {num} for key {k}!"
        self._data = data
        self._keys = data.keys()

    def get_by_idx(self, idx: int) -> Dict[str, Any]:
        """Get the prediction by specified index.

        Args:
            idx (int): The index to get predictions from.

        Returns:
            Dict[str, Any]: A dictionary with the same keys as the input data, but with the values at the specified index.
        """
        return {key: self._data[key][idx] for key in self._keys}


class BasicPredictor(
    BasePredictor,
    metaclass=AutoRegisterABCMetaClass,
):
    """BasicPredictor."""

    __is_base = True

    def __init__(
        self,
        model_dir: str,
        config: Dict[str, Any] = None,
        device: str = None,
        pp_option: PaddlePredictorOption = None,
    ) -> None:
        """Initializes the BasicPredictor.

        Args:
            model_dir (str): The directory where the model files are stored.
            config (Dict[str, Any], optional): The configuration dictionary. Defaults to None.
            device (str, optional): The device to run the inference engine on. Defaults to None.
            pp_option (PaddlePredictorOption, optional): The inference engine options. Defaults to None.
        """
        super().__init__(model_dir=model_dir, config=config)
        if not pp_option:
            pp_option = PaddlePredictorOption(model_name=self.model_name)
        if device:
            pp_option.device = device
        self.pp_option = pp_option
        self.batch_sampler = self._build_batch_sampler()
        self.result_class = self._get_result_class()
        logging.debug(f"{self.__class__.__name__}: {self.model_dir}")
        self.benchmark = benchmark

    def __call__(self, input: Any, **kwargs: Dict[str, Any]) -> Iterator[Any]:
        """
        Predict with the input data.

        Args:
            input (Any): The input data to be predicted.
            **kwargs (Dict[str, Any]): Additional keyword arguments to set up predictor.

        Returns:
            Iterator[Any]: An iterator yielding the prediction output.
        """
        self.set_predictor(**kwargs)
        if self.benchmark:
            self.benchmark.start()
            if INFER_BENCHMARK_WARMUP > 0:
                output = self.apply(input)
                warmup_num = 0
                for _ in range(INFER_BENCHMARK_WARMUP):
                    try:
                        next(output)
                        warmup_num += 1
                    except StopIteration:
                        logging.warning(
                            f"There are only {warmup_num} batches in input data, but `INFER_BENCHMARK_WARMUP` has been set to {INFER_BENCHMARK_WARMUP}."
                        )
                        break
                self.benchmark.warmup_stop(warmup_num)
            output = list(self.apply(input))
            self.benchmark.collect(len(output))
        else:
            yield from self.apply(input)

    def apply(self, input: Any) -> Iterator[Any]:
        """
        Do predicting with the input data and yields predictions.

        Args:
            input (Any): The input data to be predicted.

        Yields:
            Iterator[Any]: An iterator yielding prediction results.
        """
        for batch_data in self.batch_sampler(input):
            prediction = self.process(batch_data)
            prediction = PredictionWrap(prediction, len(batch_data))
            for idx in range(len(batch_data)):
                yield self.result_class(prediction.get_by_idx(idx))

    def set_predictor(
        self,
        batch_size: int = None,
        device: str = None,
        pp_option: PaddlePredictorOption = None,
    ) -> None:
        """
        Sets the predictor configuration.

        Args:
            batch_size (int, optional): The batch size to use. Defaults to None.
            device (str, optional): The device to run the predictor on. Defaults to None.
            pp_option (PaddlePredictorOption, optional): The predictor options to set. Defaults to None.

        Returns:
            None
        """
        if batch_size:
            self.batch_sampler.batch_size = batch_size
            self.pp_option.batch_size = batch_size
        if device and device != self.pp_option.device:
            self.pp_option.device = device
        if pp_option and pp_option != self.pp_option:
            self.pp_option = pp_option

    @abstractmethod
    def _build_batch_sampler(self) -> BaseBatchSampler:
        """Build batch sampler.

        Returns:
            BaseBatchSampler: batch sampler object.
        """
        raise NotImplementedError

    @abstractmethod
    def process(self, batch_data: List[Any]) -> Dict[str, List[Any]]:
        """process the batch data sampled from BatchSampler and return the prediction result.

        Args:
            batch_data (List[Any]): The batch data sampled from BatchSampler.

        Returns:
            Dict[str, List[Any]]: The prediction result.
        """
        raise NotImplementedError

    @abstractmethod
    def _get_result_class(self) -> type:
        """Get the result class.

        Returns:
            type: The result class.
        """
        raise NotImplementedError
