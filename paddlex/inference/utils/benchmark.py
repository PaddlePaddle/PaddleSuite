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

import functools
from types import GeneratorType
import time
import numpy as np
from prettytable import PrettyTable

from ...utils.flags import INFER_BENCHMARK_OUTPUT
from ...utils import logging


class Benchmark:
    def __init__(self, components):
        self._components = components

    def reset(self):
        for name in self._components:
            cmp = self._components[name]
            cmp.timer.reset()

    def gather(self):
        # lazy import for avoiding circular import
        from ..components.paddle_predictor import BasePaddlePredictor

        detail = []
        summary = {"preprocess": 0, "inference": 0, "postprocess": 0}
        op_tag = "preprocess"
        for name in self._components:
            cmp = self._components[name]
            times = cmp.timer.logs
            counts = len(times)
            avg = np.mean(times) * 1000
            detail.append((name, counts, avg))
            if isinstance(cmp, BasePaddlePredictor):
                summary["inference"] += avg
                op_tag = "postprocess"
            else:
                summary[op_tag] += avg
        return detail, summary

    def collect(self):
        detail, summary = self.gather()
        table = PrettyTable(["Component", "Counts", "Average Time(ms)"])
        table.add_rows([(name, cnts, f"{avg:.8f}") for name, cnts, avg in detail])
        table.add_row(("***************", "******", "***************"))
        table.add_row(("PreProcess", "\\", f"{summary['preprocess']:.8f}"))
        table.add_row(("Inference", "\\", f"{summary['inference']:.8f}"))
        table.add_row(("PostProcess", "\\", f"{summary['postprocess']:.8f}"))
        logging.info(table)

        if INFER_BENCHMARK_OUTPUT:
            str_ = "Component, Counts, Average Time(ms)\n"
            str_ += "\n".join(
                [f"{name}, {cnts}, {avg:.18f}" for name, cnts, avg in detail]
            )
            str_ += "\n***************, ***, ***************\n"
            str_ += "\n".join(
                [
                    f"PreProcess, \, {summary['preprocess']:.18f}",
                    f"Inference, \, {summary['inference']:.18f}",
                    f"PostProcess, \, {summary['postprocess']:.18f}",
                ]
            )
            with open(INFER_BENCHMARK_OUTPUT, "w") as f:
                f.write(str_)


class Timer:
    def __init__(self):
        self._tic = None
        self._elapses = []

    def watch_func(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tic = time.time()
            output = func(*args, **kwargs)
            if isinstance(output, GeneratorType):
                return self.watch_generator(output)
            else:
                self._update(time.time() - tic)
                return output

        return wrapper

    def watch_generator(self, generator):
        @functools.wraps(generator)
        def wrapper():
            while 1:
                try:
                    tic = time.time()
                    item = next(generator)
                    self._update(time.time() - tic)
                    yield item
                except StopIteration:
                    break

        return wrapper()

    def reset(self):
        self._tic = None
        self._elapses = []

    def _update(self, elapse):
        self._elapses.append(elapse)

    @property
    def logs(self):
        return self._elapses
