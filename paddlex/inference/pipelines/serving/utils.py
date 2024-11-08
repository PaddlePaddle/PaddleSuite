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

import asyncio
import base64
import io
import os
import re
import uuid
from functools import partial
from typing import Awaitable, Callable, List, Literal, Optional, TypeVar, Final, Tuple
from urllib.parse import parse_qs, urlparse

import aiohttp
import cv2
import fitz
import numpy as np
import pandas as pd
import yarl
from PIL import Image
from typing_extensions import ParamSpec, assert_never

FileType = Literal["IMAGE", "PDF"]

_P = ParamSpec("_P")
_R = TypeVar("_R")


def generate_log_id() -> str:
    return str(uuid.uuid4())


def generate_request_id() -> str:
    return str(uuid.uuid4())


def is_url(s: str) -> bool:
    if not (s.startswith("http://") or s.startswith("https://")):
        # Quick rejection
        return False
    result = urlparse(s)
    return all([result.scheme, result.netloc]) and result.scheme in ("http", "https")


def infer_file_type(url: str) -> FileType:
    # Is it more reliable to guess the file type based on the response headers?
    SUPPORTED_IMG_EXTS: Final[List[str]] = [".jpg", ".jpeg", ".png"]

    url_parts = urlparse(url)
    ext = os.path.splitext(url_parts.path)[1]
    # HACK: The support for BOS URLs with query params is implementation-based,
    # not interface-based.
    is_bos_url = (
        re.fullmatch(r"(?:bj|bd|su|gz|cd|hkg|fwh|fsh)\.bcebos\.com", url_parts.netloc)
        is not None
    )
    if is_bos_url and url_parts.query:
        params = parse_qs(url_parts.query)
        if (
            "responseContentDisposition" not in params
            or len(params["responseContentDisposition"]) != 1
        ):
            raise ValueError("`responseContentDisposition` not found")
        match_ = re.match(
            r"attachment;filename=(.*)", params["responseContentDisposition"][0]
        )
        if not match_ or not match_.groups()[0] is not None:
            raise ValueError(
                "Failed to extract the filename from `responseContentDisposition`"
            )
        ext = os.path.splitext(match_.groups()[0])[1]
    ext = ext.lower()
    if ext == ".pdf":
        return "PDF"
    elif ext in SUPPORTED_IMG_EXTS:
        return "IMAGE"
    else:
        raise ValueError("Unsupported file type")


async def get_raw_bytes(file: str, session: aiohttp.ClientSession) -> bytes:
    if is_url(file):
        async with session.get(yarl.URL(file, encoded=True)) as resp:
            return await resp.read()
    else:
        return base64.b64decode(file)


def image_bytes_to_array(data: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def image_to_base64(image: Image.Image) -> str:
    with io.BytesIO() as f:
        image.save(f, format="JPEG")
        image_base64 = base64.b64encode(f.getvalue()).decode("ascii")
    return image_base64


def csv_bytes_to_data_frame(data: bytes) -> pd.DataFrame:
    with io.StringIO(data.decode("utf-8")) as f:
        df = pd.read_csv(f)
    return df


def data_frame_to_base64(df: str) -> str:
    return base64.b64encode(df.to_csv().encode("utf-8")).decode("ascii")


def read_pdf(
    bytes_: bytes, resize: bool = False, max_num_imgs: Optional[int] = None
) -> List[np.ndarray]:
    images: List[np.ndarray] = []
    img_size = None
    with fitz.open("pdf", bytes_) as doc:
        for page in doc:
            if max_num_imgs is not None and len(images) >= max_num_imgs:
                break
            # TODO: Do not always use zoom=2.0
            zoom = 2.0
            deg = 0
            mat = fitz.Matrix(zoom, zoom).prerotate(deg)
            pixmap = page.get_pixmap(matrix=mat, alpha=False)
            image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
                pixmap.h, pixmap.w, pixmap.n
            )
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if resize:
                if img_size is None:
                    img_size = (image.shape[1], image.shape[0])
                else:
                    if (image.shape[1], image.shape[0]) != img_size:
                        image = cv2.resize(image, img_size)
            images.append(image)
    return images


def file_to_images(
    file_bytes: bytes,
    file_type: Literal["IMAGE", "PDF"],
    *,
    max_img_size: Tuple[int, int],
    max_num_imgs: int,
) -> List[np.ndarray]:
    if file_type == "IMAGE":
        images = [image_bytes_to_array(file_bytes)]
    elif file_type == "PDF":
        images = read_pdf(file_bytes, resize=True, max_num_imgs=max_num_imgs)
    else:
        assert_never(file_type)
    h, w = images[0].shape[0:2]
    if w > max_img_size[1] or h > max_img_size[0]:
        if w / h > max_img_size[0] / max_img_size[1]:
            factor = max_img_size[0] / w
        else:
            factor = max_img_size[1] / h
        images = [cv2.resize(img, (int(factor * w), int(factor * h))) for img in images]
    return images


def call_async(
    func: Callable[_P, _R], /, *args: _P.args, **kwargs: _P.kwargs
) -> Awaitable[_R]:
    return asyncio.get_running_loop().run_in_executor(
        None, partial(func, *args, **kwargs)
    )
