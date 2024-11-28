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

from .base import BaseRetriever
import os

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.embeddings import QianfanEmbeddingsEndpoint
from langchain_community.vectorstores import FAISS
from langchain_community import vectorstores
from erniebot_agent.extensions.langchain.embeddings import ErnieEmbeddings

import time


class ErnieBotRetriever(BaseRetriever):
    """Ernie Bot Retriever"""

    entities = [
        "ernie-4.0",
        "ernie-3.5",
        "ernie-3.5-8k",
        "ernie-lite",
        "ernie-tiny-8k",
        "ernie-speed",
        "ernie-speed-128k",
        "ernie-char-8k",
    ]

    def __init__(self, config):

        super().__init__()

        model_name = config.get("model_name", None)
        api_type = config.get("api_type", None)
        ak = config.get("ak", None)
        sk = config.get("sk", None)
        access_token = config.get("access_token", None)

        if model_name not in self.entities:
            raise ValueError(f"model_name must be in {self.entities} of ErnieBotChat.")

        if api_type not in ["aistudio", "qianfan"]:
            raise ValueError("api_type must be one of ['aistudio', 'qianfan']")

        if api_type == "aistudio" and access_token is None:
            raise ValueError("access_token cannot be empty when api_type is aistudio.")

        if api_type == "qianfan" and (ak is None or sk is None):
            raise ValueError("ak and sk cannot be empty when api_type is qianfan.")

        self.model_name = model_name
        self.config = config

    def generate_vector_database(
        self,
        text_list,
        block_size=300,
        separators=["\t", "\n", "。", "\n\n", ""],
        sleep_time=0.5,
    ):
        """
        args:
        return:
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=block_size, chunk_overlap=20, separators=separators
        )
        texts = text_splitter.split_text("\t".join(text_list))
        all_splits = [Document(page_content=text) for text in texts]

        api_type = self.config["api_type"]
        if api_type == "qianfan":
            os.environ["QIANFAN_AK"] = os.environ.get("EB_AK", self.config["ak"])
            os.environ["QIANFAN_SK"] = os.environ.get("EB_SK", self.config["sk"])
            user_ak = os.environ.get("EB_AK", self.config["ak"])
            user_id = hash(user_ak)
            vectorstore = FAISS.from_documents(
                documents=all_splits, embedding=QianfanEmbeddingsEndpoint()
            )
        elif api_type == "aistudio":
            token = self.config["access_token"]
            vectorstore = FAISS.from_documents(
                documents=all_splits[0:1],
                embedding=ErnieEmbeddings(aistudio_access_token=token),
            )
            #### ErnieEmbeddings.chunk_size = 16
            step = min(16, len(all_splits) - 1)
            for shot_splits in [
                all_splits[i : i + step] for i in range(1, len(all_splits), step)
            ]:
                time.sleep(sleep_time)
                vectorstore_slice = FAISS.from_documents(
                    documents=shot_splits,
                    embedding=ErnieEmbeddings(aistudio_access_token=token),
                )
                vectorstore.merge_from(vectorstore_slice)
        else:
            raise ValueError(f"Unsupported api_type: {api_type}")

        return vectorstore

    def encode_vector_store_to_bytes(self, vectorstore):
        vectorstore = self.encode_vector_store(vectorstore.serialize_to_bytes())
        return vectorstore

    def decode_vector_store_from_bytes(self, vectorstore):
        if not self.is_vector_store(vectorstore):
            raise ValueError("The retrieved vectorstore is not for PaddleX.")
        api_type = self.config["api_type"]

        if api_type == "aistudio":
            access_token = self.config["access_token"]
            embeddings = ErnieEmbeddings(aistudio_access_token=access_token)
        elif api_type == "qianfan":
            ak = self.config["ak"]
            sk = self.config["sk"]
            embeddings = QianfanEmbeddingsEndpoint(qianfan_ak=ak, qianfan_sk=sk)
        else:
            raise ValueError(f"Unsupported api_type: {api_type}")
        vector = vectorstores.FAISS.deserialize_from_bytes(
            self.decode_vector_store(vectorstore), embeddings
        )
        return vector

    def similarity_retrieval(self, query_text_list, vectorstore, sleep_time=0.5):
        # 根据提问匹配上下文
        C = []
        for query_text in query_text_list:
            QUESTION = query_text
            time.sleep(sleep_time)
            docs = vectorstore.similarity_search_with_relevance_scores(QUESTION, k=2)
            context = [(document.page_content, score) for document, score in docs]
            context = sorted(context, key=lambda x: x[1])
            C.extend([x[0] for x in context[::-1]])
        C = list(set(C))
        all_C = " ".join(C)
        return all_C
