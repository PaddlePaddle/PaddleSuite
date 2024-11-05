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
        self._e2e_tic = None
        self._e2e_elapse = None

    def reset(self):
        for name in self._components:
            cmp = self._components[name]
            cmp.timer.reset()
        self._e2e_tic = time.time()

    def gather(self, e2e_num):
        # lazy import for avoiding circular import
        from ..components.paddle_predictor import BasePaddlePredictor

        detail = []
        summary = {"preprocess": 0, "inference": 0, "postprocess": 0}
        op_tag = "preprocess"
        for name in self._components:
            cmp = self._components[name]
            times = cmp.timer.logs
            counts = len(times)
            avg = np.mean(times)
            total = np.sum(times)
            detail.append((name, counts, avg))
            if isinstance(cmp, BasePaddlePredictor):
                summary["inference"] += total
                op_tag = "postprocess"
            else:
                summary[op_tag] += total

        summary = [
            ("PreProcess", e2e_num, summary["preprocess"] / e2e_num),
            ("Inference", e2e_num, summary["inference"] / e2e_num),
            ("PostProcess", e2e_num, summary["postprocess"] / e2e_num),
            ("End2End", e2e_num, self._e2e_elapse / e2e_num),
        ]
        return detail, summary

    def collect(self, e2e_num):
        self._e2e_elapse = time.time() - self._e2e_tic
        detail, summary = self.gather(e2e_num)

        table = PrettyTable(["Component", "Call Counts", "Avg Time Per Call (ms)"])
        table.add_rows(
            [(name, cnts, f"{avg * 1000:.8f}") for name, cnts, avg in detail]
        )
        logging.info(table)

        table = PrettyTable(["Stage", "Num of Instances", "Avg Time Per Instance (ms)"])
        table.add_rows(
            [(name, cnts, f"{avg * 1000:.8f}") for name, cnts, avg in summary]
        )
        logging.info(table)

        if INFER_BENCHMARK_OUTPUT:
            str_ = "Component, Call Counts, Avg Time Per Call (ms)\n"
            str_ += "\n".join(
                [f"{name}, {cnts}, {avg * 1000:.18f}" for name, cnts, avg in detail]
            )
            str_ += "\n" + "*" * 100 + "\n"
            str_ += "Stage, Num of Instances, Avg Time Per Instance (ms)\n"
            str_ += "\n".join(
                [f"{name}, {cnts}, {avg * 1000:.18f}" for name, cnts, avg in summary]
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
