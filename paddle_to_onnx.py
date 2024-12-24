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

import argparse
import sys
import subprocess
import shutil
from pathlib import Path


PD_MODEL_FILENAME = "inference.pdmodel"
PD_PARAMS_FILENAME = "inference.pdiparams"
ONNX_MODEL_FILENAME = "inference.onnx"
CONFIG_FILENAME = "inference.yml"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Input model directory."
    )
    parser.add_argument(
        "--output_dir", type=str, default="onnx", help="Output model directory."
    )
    parser.add_argument("--opset_version", type=int, default=9)
    args = parser.parse_args()
    return args


def check_input_dir(input_dir):
    input_dir = Path(input_dir)
    if not input_dir.exists():
        sys.exit(f"'{input_dir}' does not exist")
    if not input_dir.is_dir():
        sys.exit(f"'{input_dir}' is not a directory")
    model_path = input_dir / PD_MODEL_FILENAME
    if not model_path.exists():
        sys.exit(f"'{model_path}' does not exist")
    params_path = input_dir / PD_PARAMS_FILENAME
    if not params_path.exists():
        sys.exit(f"'{params_path}' does not exist")
    config_path = input_dir / CONFIG_FILENAME
    if not config_path.exists():
        sys.exit(f"'{config_path}' does not exist")


def check_paddle2onnx():
    if shutil.which("paddle2onnx") is None:
        sys.exit("Paddle2ONNX not available")


def run_paddle2onnx(input_dir, output_dir, opset_version):
    print("Paddle2ONNX conversion starting...")
    cmd = [
        "paddle2onnx",
        "--model_dir",
        input_dir,
        "--model_filename",
        PD_MODEL_FILENAME,
        "--params_filename",
        PD_PARAMS_FILENAME,
        "--save_file",
        str(Path(output_dir, ONNX_MODEL_FILENAME)),
        "--opset_version",
        str(opset_version),
    ]
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        sys.exit(f"Paddle2ONNX conversion failed with exit code {e.returncode}")
    print("Paddle2ONNX conversion succeeded")


def copy_config_file(input_dir, output_dir):
    src_path = Path(input_dir, CONFIG_FILENAME)
    dst_path = Path(output_dir, CONFIG_FILENAME)
    shutil.copy(src_path, dst_path)
    print(f"Copied '{src_path}' to '{dst_path}'")


if __name__ == "__main__":
    args = parse_args()
    print(f"Input dir: {args.input_dir}")
    print(f"Output dir: {args.output_dir}")
    check_input_dir(args.input_dir)
    check_paddle2onnx()
    run_paddle2onnx(args.input_dir, args.output_dir, args.opset_version)
    copy_config_file(args.input_dir, args.output_dir)
    print("Done")
