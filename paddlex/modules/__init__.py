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


from .base import (
    build_dataset_checker,
    build_trainer,
    build_evaluater,
    build_exportor,
    build_predictor,
    create_model,
    PaddleInferenceOption,
)
from .image_classification import (
    ClsDatasetChecker,
    ClsTrainer,
    ClsEvaluator,
    ClsExportor,
    ClsPredictor,
)
from .object_detection import (
    COCODatasetChecker,
    DetTrainer,
    DetEvaluator,
    DetExportor,
    DetPredictor,
)
from .text_detection import (
    TextDetDatasetChecker,
    TextDetTrainer,
    TextDetEvaluator,
    TextDetExportor,
    TextDetPredictor,
)
from .text_recognition import (
    TextRecDatasetChecker,
    TextRecTrainer,
    TextRecEvaluator,
    TextRecExportor,
    TextRecPredictor,
)
from .table_recognition import (
    TableRecDatasetChecker,
    TableRecTrainer,
    TableRecEvaluator,
    TableRecExportor,
    TableRecPredictor,
)
from .semantic_segmentation import (
    SegDatasetChecker,
    SegTrainer,
    SegEvaluator,
    SegExportor,
    SegPredictor,
)
from .instance_segmentation import (
    COCOInstSegDatasetChecker,
    InstanceSegTrainer,
    InstanceSegEvaluator,
    InstanceSegExportor,
    InstanceSegPredictor,
)
from .ts_anomaly_detection import (
    TSADDatasetChecker,
    TSADTrainer,
    TSADEvaluator,
    TSADPredictor,
)
from .ts_classification import (
    TSCLSDatasetChecker,
    TSCLSTrainer,
    TSCLSEvaluator,
    TSCLSPredictor,
)
from .ts_forecast import TSFCDatasetChecker, TSFCTrainer, TSFCEvaluator, TSFCPredictor

from .base.predictor.transforms import image_common
from .image_classification import transforms as cls_transforms
from .object_detection import transforms as det_transforms
from .text_detection import transforms as text_det_transforms
from .text_recognition import transforms as text_rec_transforms
from .table_recognition import transforms as table_rec_transforms
from .semantic_segmentation import transforms as seg_transforms
from .instance_segmentation import transforms as instance_seg_transforms
