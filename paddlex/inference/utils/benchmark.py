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

import csv
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
        self._warmup_start = None
        self._warmup_elapse = None
        self._warmup_num = None
        self._e2e_tic = None
        self._e2e_elapse = None

    def start(self):
        self._warmup_start = time.time()
        self._reset()

    def warmup_stop(self, warmup_num):
        self._warmup_elapse = time.time() - self._warmup_start
        self._warmup_num = warmup_num
        self._reset()

    def _reset(self):
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
            detail.append((name, total, counts, avg))
            if isinstance(cmp, BasePaddlePredictor):
                summary["inference"] += total
                op_tag = "postprocess"
            else:
                summary[op_tag] += total

        summary = [
            (
                "PreProcess",
                summary["preprocess"],
                e2e_num,
                summary["preprocess"] / e2e_num,
            ),
            (
                "Inference",
                summary["inference"],
                e2e_num,
                summary["inference"] / e2e_num,
            ),
            (
                "PostProcess",
                summary["postprocess"],
                e2e_num,
                summary["postprocess"] / e2e_num,
            ),
            ("End2End", self._e2e_elapse, e2e_num, self._e2e_elapse / e2e_num),
        ]
        if self._warmup_elapse:
            summary.append(
                (
                    "WarmUp",
                    self._warmup_elapse,
                    self._warmup_num,
                    self._warmup_elapse / self._warmup_num,
                )
            )
        return detail, summary

    def collect(self, e2e_num):
        self._e2e_elapse = time.time() - self._e2e_tic
        detail, summary = self.gather(e2e_num)

        table_head = ["Stage", "Total Time (ms)", "Nums", "Avg Time (ms)"]
        table = PrettyTable(table_head)
        table.add_rows(
            [
                (name, f"{total * 1000:.8f}", cnts, f"{avg * 1000:.8f}")
                for name, total, cnts, avg in detail
            ]
        )
        logging.info(table)

        table = PrettyTable(table_head)
        table.add_rows(
            [
                (name, f"{total * 1000:.8f}", cnts, f"{avg * 1000:.8f}")
                for name, total, cnts, avg in summary
            ]
        )
        logging.info(table)

        if INFER_BENCHMARK_OUTPUT:
            csv_data = [table_head]
            csv_data.extend(detail)
            csv_data.extend(summary)
            with open("benchmark.csv", "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerows(csv_data)


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
