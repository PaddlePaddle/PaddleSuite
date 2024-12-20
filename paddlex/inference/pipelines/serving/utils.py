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
from typing import (
    Awaitable,
    Callable,
    List,
    Literal,
    Optional,
    TypeVar,
    Final,
    Tuple,
    overload,
    Union,
)
from urllib.parse import parse_qs, urlparse

import aiohttp
import cv2
import fitz
import numpy as np
import pandas as pd
import yarl
from PIL import Image
from typing_extensions import ParamSpec, assert_never

from .models import ImageInfo, PDFInfo, PDFPageInfo

FileType = Literal["IMAGE", "PDF"]

_P = ParamSpec("_P")
_R = TypeVar("_R")


def generate_log_id() -> str:
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


def image_bytes_to_image(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data))


def image_to_bytes(image: Image.Image, format: str = "JPEG") -> bytes:
    with io.BytesIO() as f:
        image.save(f, format=format)
        img_bytes = f.getvalue()
    return img_bytes


def image_array_to_bytes(image: np.ndarray, ext: str = ".jpg") -> bytes:
    image = cv2.imencode(ext, image)[1]
    return image.tobytes()


def csv_bytes_to_data_frame(data: bytes) -> pd.DataFrame:
    with io.StringIO(data.decode("utf-8")) as f:
        df = pd.read_csv(f)
    return df


def data_frame_to_bytes(df: str) -> str:
    return df.to_csv().encode("utf-8")


def base64_encode(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def read_pdf(
    bytes_: bytes, max_num_imgs: Optional[int] = None
) -> Tuple[List[np.ndarray], PDFInfo]:
    images: List[np.ndarray] = []
    page_info_list: List[PDFPageInfo] = []
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
            images.append(image)
            page_info = PDFPageInfo(
                width=pixmap.w,
                height=pixmap.h,
            )
            page_info_list.append(page_info)
    pdf_info = PDFInfo(
        numPages=len(page_info_list),
        pages=page_info_list,
    )
    return images, pdf_info


@overload
def file_to_images(
    file_bytes: bytes,
    file_type: Literal["IMAGE"],
    *,
    max_num_imgs: Optional[int] = ...,
) -> Tuple[List[np.ndarray], ImageInfo]: ...


@overload
def file_to_images(
    file_bytes: bytes,
    file_type: Literal["PDF"],
    *,
    max_num_imgs: Optional[int] = ...,
) -> Tuple[List[np.ndarray], PDFInfo]: ...


def file_to_images(
    file_bytes: bytes,
    file_type: Literal["IMAGE", "PDF"],
    *,
    max_num_imgs: Optional[int] = None,
) -> Tuple[List[np.ndarray], Union[ImageInfo, PDFInfo]]:
    if file_type == "IMAGE":
        images = [image_bytes_to_array(file_bytes)]
        data_info = get_image_info(images[0])
    elif file_type == "PDF":
        images, data_info = read_pdf(file_bytes, max_num_imgs=max_num_imgs)
    else:
        assert_never(file_type)
    return images, data_info


def get_image_info(image: np.ndarray) -> ImageInfo:
    return ImageInfo(width=image.shape[1], height=image.shape[0])


def call_async(
    func: Callable[_P, _R], /, *args: _P.args, **kwargs: _P.kwargs
) -> Awaitable[_R]:
    return asyncio.get_running_loop().run_in_executor(
        None, partial(func, *args, **kwargs)
    )
