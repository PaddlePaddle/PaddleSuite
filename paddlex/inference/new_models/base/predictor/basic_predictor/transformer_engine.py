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

import networkx as nx

from ......utils import logging
from ...batch_sampler import BaseBatchSampler


class TransformerEngine(object):
    def __init__(self, cmpts):
        self._cmpts = cmpts
        self.keys = list(cmpts.keys())
        # graph = self._build_graph()

    def _build_graph(self):
        graph = nx.DiGraph()
        for name in self._cmpts:
            cmpt = self._cmpts[name]
            graph.add_node(name, transform=cmpt)
            for in_param, out_param in cmpt.dependencies:
                logging.debug(f"{in_param} <-- {out_param}")
                graph.add_edge(out_param.cmpt.name, in_param.cmpt.name)

        execution_order = list(nx.topological_sort(graph))
        logging.debug(f"Execution Order: {execution_order}")
        return graph

    def __call__(self, data, i=0):
        data = self._cmpts[self.keys[i]](data)
        if i + 1 < len(self._cmpts):
            return self.__call__(data, i + 1)
        else:
            return data
