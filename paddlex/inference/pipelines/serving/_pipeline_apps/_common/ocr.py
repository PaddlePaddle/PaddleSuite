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

from typing import Final, List, Literal, Optional, Tuple

import numpy as np
from fastapi import HTTPException
from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypeAlias

from ......utils import logging
from ... import utils as serving_utils
from ...app import AppContext

DEFAULT_MAX_IMG_SIZE: Final[Tuple[int, int]] = (2000, 2000)
DEFAULT_MAX_NUM_IMGS: Final[int] = 10

FileType: TypeAlias = Literal[0, 1]


class InferenceParams(BaseModel):
    maxLongSide: Optional[Annotated[int, Field(gt=0)]] = None


class InferRequest(BaseModel):
    file: str
    fileType: Optional[FileType] = None
    inferenceParams: Optional[InferenceParams] = None


def update_app_context(app_context: AppContext) -> None:
    cfg = app_context.config.extra or {}
    app_context.extra["max_img_size"] = cfg.get("max_img_size", DEFAULT_MAX_IMG_SIZE)
    app_context.extra["max_num_imgs"] = cfg.get("max_num_imgs", DEFAULT_MAX_NUM_IMGS)


def get_file_type(request: InferRequest) -> Literal["IMAGE", "PDF"]:
    if request.fileType is None:
        if serving_utils.is_url(request.file):
            try:
                file_type = serving_utils.infer_file_type(request.file)
            except Exception:
                logging.exception("Failed to infer the file type")
                raise HTTPException(
                    status_code=422,
                    detail="The file type cannot be inferred from the URL. Please specify the file type explicitly.",
                )
        else:
            raise HTTPException(status_code=422, detail="Unknown file type")
    else:
        file_type = "PDF" if request.fileType == 0 else "IMAGE"

    return file_type


async def get_images(
    request: InferRequest, app_context: AppContext
) -> List[np.ndarray]:
    file_type = get_file_type(request)
    # XXX: Currently, we use 500 for consistency. However, 422 may be more
    # appropriate.
    try:
        file_bytes = await serving_utils.get_raw_bytes(
            request.file,
            app_context.aiohttp_session,
        )
        return await serving_utils.call_async(
            serving_utils.file_to_images,
            file_bytes,
            file_type,
            max_img_size=app_context.extra["max_img_size"],
            max_num_imgs=app_context.extra["max_num_imgs"],
        )
    except Exception:
        logging.exception("Unexpected exception")
        raise HTTPException(status_code=500, detail="Internal server error")
