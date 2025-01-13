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
import mimetypes
import tempfile
import uuid
from functools import partial
from typing import Awaitable, Callable, List, Optional, Tuple, TypeVar, Union, overload
from urllib.parse import urlparse

import aiohttp
import cv2
import filetype
import fitz
import numpy as np
import pandas as pd
import yarl
from PIL import Image
from typing_extensions import Literal, ParamSpec, TypeAlias, assert_never

from ._models import ImageInfo, PDFInfo, PDFPageInfo

FileType: TypeAlias = Literal["IMAGE", "PDF", "VIDEO", "AUDIO"]

_P = ParamSpec("_P")
_R = TypeVar("_R")


def generate_log_id() -> str:
    return str(uuid.uuid4())


# TODO:
# 1. Use Pydantic to validate the URL and Base64-encoded string types for both
#    input and output data instead of handling this manually.
# 2. Define a `File` type for global use; this will be part of the contract.
# 3. Consider using two separate fields instead of a union of URL and Base64,
#    even though they are both strings. Backward compatibility should be
#    maintained.
def is_url(s: str) -> bool:
    if not (s.startswith("http://") or s.startswith("https://")):
        # Quick rejection
        return False
    result = urlparse(s)
    return all([result.scheme, result.netloc]) and result.scheme in ("http", "https")


def infer_file_type(url: str) -> Optional[FileType]:
    url_parts = urlparse(url)
    filename = url_parts.path.split("/")[-1]

    file_type = mimetypes.guess_type(filename)[0]

    if file_type is None:
        return None

    if file_type.startswith("image/"):
        return "IMAGE"
    elif file_type == "application/pdf":
        return "PDF"
    elif file_type.startswith("video/"):
        return "VIDEO"
    elif file_type.startswith("audio/"):
        return "AUDIO"
    else:
        return None


def infer_file_ext(file: str) -> Optional[str]:
    if is_url(file):
        url_parts = urlparse(file)
        filename = url_parts.path.split("/")[-1]
        mime_type = mimetypes.guess_type(filename)[0]
        if mime_type is None:
            return None
        return mimetypes.guess_extension(mime_type)
    else:
        bytes_ = base64.b64decode(file)
        return filetype.guess_extension(bytes_)


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


def data_frame_to_bytes(df: pd.DataFrame) -> bytes:
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


@overload
def file_to_images(
    file_bytes: bytes,
    file_type: Literal["IMAGE", "PDF"],
    *,
    max_num_imgs: Optional[int] = ...,
) -> Union[Tuple[List[np.ndarray], ImageInfo], Tuple[List[np.ndarray], PDFInfo]]: ...


def file_to_images(
    file_bytes: bytes,
    file_type: Literal["IMAGE", "PDF"],
    *,
    max_num_imgs: Optional[int] = None,
) -> Union[Tuple[List[np.ndarray], ImageInfo], Tuple[List[np.ndarray], PDFInfo]]:
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


def write_to_temp_file(file_bytes: bytes, suffix: str) -> str:
    with tempfile.NamedTemporaryFile("wb", suffix=suffix, delete=False) as f:
        f.write(file_bytes)
        return f.name


async def get_raw_bytes(file: str, session: aiohttp.ClientSession) -> bytes:
    if is_url(file):
        async with session.get(yarl.URL(file, encoded=True)) as resp:
            return await resp.read()
    else:
        return base64.b64decode(file)


def call_async(
    func: Callable[_P, _R], /, *args: _P.args, **kwargs: _P.kwargs
) -> Awaitable[_R]:
    return asyncio.get_running_loop().run_in_executor(
        None, partial(func, *args, **kwargs)
    )
