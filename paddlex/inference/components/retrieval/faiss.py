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

import os
import pickle
from pathlib import Path
import faiss
import numpy as np

from ....utils import logging
from ..base import BaseComponent


class IndexData:
    def __init__(self, index, id_map):
        self._index = index
        self._id_map = id_map

    @property
    def index(self):
        return self._index

    @property
    def index_bytes(self):
        return faiss.serialize_index(self._index)

    @property
    def id_map(self):
        return self._id_map

    def save(self, save_path):
        index_data = {
            "index_bytes": self.index_bytes,
            "id_map": self.id_map,
        }
        with open(save_path, "wb") as fd:
            pickle.dump(index_data, fd)

    @classmethod
    def load(self, index):
        if isinstance(index, str):
            with open(index, "rb") as fd:
                index_data = pickle.load(fd)
            index_ = faiss.deserialize_index(index_data["index_bytes"])
            id_map = index_data["id_map"]
            assert index_.ntotal == len(
                id_map
            ), "data number in index is not equal in in id_map"
            return index_, id_map
        else:
            assert isinstance(index, IndexData)
            return index.index, index.id_map


class FaissIndexer(BaseComponent):

    INPUT_KEYS = "feature"
    OUTPUT_KEYS = ["label", "score"]
    DEAULT_INPUTS = {"feature": "feature"}
    DEAULT_OUTPUTS = {"label": "label", "score": "score"}

    ENABLE_BATCH = True

    def __init__(
        self,
        index,
        metric_type="IP",
        return_k=1,
        score_thres=None,
        hamming_radius=None,
    ):
        super().__init__()

        if metric_type == "hamming":
            self.hamming_radius = hamming_radius
        else:
            self.score_thres = score_thres

        self._indexer, self.id_map = IndexData.load(index)
        self.metric_type = metric_type
        self.return_k = return_k

    def apply(self, feature):
        """apply"""
        scores_list, ids_list = self._indexer.search(np.array(feature), self.return_k)
        preds = []
        for scores, ids in zip(scores_list, ids_list):
            labels = []
            for id in ids:
                if id > 0:
                    labels.append(self.id_map[id])
            preds.append({"score": scores, "label": labels})

        if self.metric_type == "hamming":
            idxs = np.where(scores_list[:, 0] > self.hamming_radius)[0]
        else:
            idxs = np.where(scores_list[:, 0] < self.score_thres)[0]
        for idx in idxs:
            preds[idx] = {"score": None, "label": None}
        return preds


class FaissBuilder:

    SUPPORT_METRIC_TYPE = ("hamming", "IP", "L2")
    SUPPORT_INDEX_TYPE = ("Flat", "IVF", "HNSW32")
    BINARY_METRIC_TYPE = ("hamming",)
    BINARY_SUPPORT_INDEX_TYPE = ("Flat", "IVF", "BinaryHash")

    @classmethod
    def _get_index_type(cls, metric_type, index_type, num=None):
        # if IVF method, cal ivf number automaticlly
        if index_type == "IVF":
            index_type = index_type + str(min(int(num // 8), 65536))
            if metric_type in cls.BINARY_METRIC_TYPE:
                index_type += ",BFlat"
            else:
                index_type += ",Flat"

        # for binary index, add B at head of index_type
        if metric_type in cls.BINARY_METRIC_TYPE:
            assert (
                index_type in cls.BINARY_SUPPORT_INDEX_TYPE
            ), f"The metric type({metric_type}) only support {cls.BINARY_SUPPORT_INDEX_TYPE} index types!"
            index_type = "B" + index_type

        if index_type == "HNSW32":
            logging.warning("The HNSW32 method dose not support 'remove' operation")
            index_type = "HNSW32"

        if index_type == "Flat":
            index_type = "Flat"

        return index_type

    @classmethod
    def _get_metric_type(cls, metric_type):
        if metric_type == "hamming":
            return faiss.METRIC_Hamming
        elif metric_type == "jaccard":
            return faiss.METRIC_Jaccard
        elif metric_type == "IP":
            return faiss.METRIC_INNER_PRODUCT
        elif metric_type == "L2":
            return faiss.METRIC_L2

    @classmethod
    def build(
        cls,
        gallery_imgs,
        gallery_label,
        predict_func,
        metric_type="IP",
        index_type="HNSW32",
    ):
        assert (
            metric_type in cls.SUPPORT_METRIC_TYPE
        ), f"Supported metric types only: {cls.SUPPORT_METRIC_TYPE}!"

        if isinstance(gallery_label, str):
            gallery_docs, gallery_list = cls.load_gallery(gallery_label, gallery_imgs)
        else:
            gallery_docs, gallery_list = gallery_label, gallery_imgs

        features = [res["feature"] for res in predict_func(gallery_list)]
        dtype = np.uint8 if metric_type in cls.BINARY_METRIC_TYPE else np.float32
        features = np.array(features).astype(dtype)
        vector_num, vector_dim = features.shape

        if metric_type in cls.BINARY_METRIC_TYPE:
            index = faiss.index_binary_factory(
                vector_dim,
                cls._get_index_type(metric_type, index_type, vector_num),
                cls._get_metric_type(metric_type),
            )
        else:
            index = faiss.index_factory(
                vector_dim,
                cls._get_index_type(metric_type, index_type, vector_num),
                cls._get_metric_type(metric_type),
            )
            index = faiss.IndexIDMap2(index)
        ids = {}

        # calculate id for new data
        index, ids = cls._add_gallery(
            metric_type, index, ids, features, gallery_docs, mode="new"
        )
        return IndexData(index, ids)

    @classmethod
    def remove(
        cls,
        gallery_label,
        index,
        index_type="HNSW32",
    ):
        assert (
            index_type in cls.SUPPORT_INDEX_TYPE
        ), f"Supported index types only: {cls.SUPPORT_INDEX_TYPE}!"

        if isinstance(gallery_label, str):
            gallery_docs, _ = cls.load_gallery(gallery_label)
        else:
            gallery_docs = gallery_label

        index, ids = IndexData.load(index)
        if index_type == "HNSW32":
            raise RuntimeError(
                "The index_type: HNSW32 dose not support 'remove' operation"
            )

        # remove ids in id_map, remove index data in faiss index
        index, ids = cls._rm_id_in_gallery(index, ids, gallery_docs)
        return IndexData(index, ids)

    @classmethod
    def append(cls, gallery_imgs, gallery_label, predict_func, index, metric_type="IP"):
        assert (
            metric_type in cls.SUPPORT_METRIC_TYPE
        ), f"Supported metric types only: {cls.SUPPORT_METRIC_TYPE}!"

        if isinstance(gallery_label, str):
            gallery_docs, gallery_list = cls.load_gallery(gallery_label, gallery_imgs)
        else:
            gallery_docs, gallery_list = gallery_label, gallery_imgs

        features = [res["feature"] for res in predict_func(gallery_list)]
        dtype = np.uint8 if metric_type in cls.BINARY_METRIC_TYPE else np.float32
        features = np.array(features).astype(dtype)

        index, ids = IndexData.load(index)

        # calculate id for new data
        index, ids = cls._add_gallery(
            metric_type, index, ids, features, gallery_docs, mode="append"
        )
        return IndexData(index, ids)

    @classmethod
    def _add_gallery(
        cls, metric_type, index, ids, gallery_features, gallery_docs, mode
    ):
        start_id = max(ids.keys()) + 1 if ids else 0
        ids_now = (np.arange(0, len(gallery_docs)) + start_id).astype(np.int64)

        # only train when new index file
        if mode == "new":
            if metric_type in cls.BINARY_METRIC_TYPE:
                index.add(gallery_features)
            else:
                index.train(gallery_features)

        if not metric_type in cls.BINARY_METRIC_TYPE:
            index.add_with_ids(gallery_features, ids_now)

        for i, d in zip(list(ids_now), gallery_docs):
            ids[i] = d
        return index, ids

    @classmethod
    def _rm_id_in_gallery(cls, index, ids, gallery_docs):
        remove_ids = list(filter(lambda k: ids.get(k) in gallery_docs, ids.keys()))
        remove_ids = np.asarray(remove_ids)
        index.remove_ids(remove_ids)
        for k in remove_ids:
            del ids[k]

        return index, ids

    @classmethod
    def load_gallery(cls, gallery_label_path, gallery_imgs_root="", delimiter=" "):
        lines = []
        files = []
        labels = []
        root = Path(gallery_imgs_root)
        with open(gallery_label_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines:
            path, label = line.strip().split(delimiter)
            file_path = root / path
            files.append(file_path.as_posix())
            labels.append(label)
        return labels, files
