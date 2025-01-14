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

import pickle
import uuid

import faiss

from .....pipelines_new.components.faisser import IndexData


# XXX: I have to implement serialization and deserialization functions myself,
# which is fragile.
def serialize_index_data(index_data: IndexData) -> bytes:
    tup = (index_data.index_bytes, index_data.index_info)
    return pickle.dumps(tup)


def deserialize_index_data(index_data_bytes: bytes) -> IndexData:
    tup = pickle.loads(index_data_bytes)
    index = faiss.deserialize_index(tup[0])
    return IndexData(index, tup[1])


def generate_index_key() -> str:
    return str(uuid.uuid4())
