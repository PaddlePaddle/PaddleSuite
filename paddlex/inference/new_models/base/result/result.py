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

import inspect

from .....utils.func_register import FuncRegister
from ....utils.io import ImageReader, ImageWriter
from .mixin import JsonMixin, ImgMixin, StrMixin


class BaseResult(dict, StrMixin, JsonMixin):
    def __init__(self, data):
        super().__init__(data)
        self._show_funcs = []
        StrMixin.__init__(self)
        JsonMixin.__init__(self)

    def save_all(self, save_path):
        for func in self._show_funcs:
            signature = inspect.signature(func)
            if "save_path" in signature.parameters:
                func(save_path=save_path)
            else:
                func()


class CVResult(BaseResult, ImgMixin):

    INPUT_KEYS = ["input_img"]

    def __init__(self, data):
        assert set(CVResult.INPUT_KEYS) <= set(
            self.INPUT_KEYS
        ), f"`{CVResult.INPUT_KEYS}` is needed, but not found in `{self.INPUT_KEYS}`!"
        self._input_img = data.pop("input_img", None)
        super().__init__(data)
        ImgMixin.__init__(self, "pillow")
        self._img_writer = ImageWriter(backend="pillow")
