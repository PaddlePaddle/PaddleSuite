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
import argparse
import textwrap
from types import SimpleNamespace

from . import create_pipeline
from .inference.pipelines import create_pipeline_from_config, load_pipeline_config
from .repo_manager import setup, get_all_supported_repo_names
from .utils import logging


def args_cfg():
    """parse cli arguments"""

    def parse_str(s):
        """convert str type value
        to None type if it is "None",
        to bool type if it means True or False.
        """
        if s in ("None"):
            return None
        elif s in ("TRUE", "True", "true", "T", "t"):
            return True
        elif s in ("FALSE", "False", "false", "F", "f"):
            return False
        return s

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd")

    ################# install pdx #################
    parser.add_argument("--install", action="store_true", default=False, help="")
    parser.add_argument("devkits", nargs="*", default=[])
    parser.add_argument("--no_deps", action="store_true")
    parser.add_argument("--platform", type=str, default="github.com")
    parser.add_argument(
        "-y",
        "--yes",
        dest="update_repos",
        action="store_true",
        help="Whether to update_repos all packages.",
    )
    parser.add_argument(
        "--use_local_repos",
        action="store_true",
        default=False,
        help="Use local repos when existing.",
    )

    ################# pipeline predict #################
    parser.add_argument("--predict", action="store_true", default=True, help="")
    parser.add_argument("--pipeline", type=str, help="")
    parser.add_argument("--model", nargs="+", help="")
    parser.add_argument("--model_dir", nargs="+", type=parse_str, help="")
    parser.add_argument("--input", type=str, help="")
    parser.add_argument("--save_dir", type=str, default="./", help="")
    parser.add_argument("--device", type=str, default="gpu:0", help="")
    parser.add_argument("--use_hpip", action="store_true")
    parser.add_argument("--serial_number", type=str)
    parser.add_argument("--update_license", action="store_true")

    ################# serving #################
    serving_parser = subparsers.add_parser("serve")
    serving_parser.add_argument("pipeline", type=str)
    serving_parser.add_argument("--device", type=str)
    serving_parser.add_argument("--use_hpip", action="store_true")
    serving_parser.add_argument("--serial_number", type=str)
    serving_parser.add_argument("--update_license", action="store_true")
    serving_parser.add_argument("--host", type=str, default="0.0.0.0")
    serving_parser.add_argument("--port", type=int, default=8080)
    serving_parser.add_argument("--debug", action="store_true")

    return parser.parse_args()


def install(args):
    """install paddlex"""
    # Enable debug info
    os.environ["PADDLE_PDX_DEBUG"] = "True"
    # Disable eager initialization
    os.environ["PADDLE_PDX_EAGER_INIT"] = "False"

    repo_names = args.devkits
    if len(repo_names) == 0:
        repo_names = get_all_supported_repo_names()
    setup(
        repo_names=repo_names,
        no_deps=args.no_deps,
        platform=args.platform,
        update_repos=args.update_repos,
        use_local_repos=args.use_local_repos,
    )
    return


def _get_hpi_params(serial_number, update_license):
    return {"serial_number": serial_number, "update_license": update_license}


def pipeline_predict(
    pipeline, input_path, device, save_dir, use_hpip, serial_number, update_license
):
    """pipeline predict"""
    hpi_params = _get_hpi_params(serial_number, update_license)
    pipeline = create_pipeline(pipeline, use_hpip=use_hpip, hpi_params=hpi_params)
    result = pipeline(input_path, device=device)
    for res in result:
        res.print(json_format=False)
        # TODO(gaotingquan): support to save all
        # if save_dir:
        #     i["result"].save()


def serve(
    pipeline, *, device, use_hpip, serial_number, update_license, host, port, debug
):
    from .inference.pipelines.serving import create_pipeline_app, run_server

    hpi_params = _get_hpi_params(serial_number, update_license)
    pipeline_config = load_pipeline_config(pipeline)
    pipeline = create_pipeline_from_config(
        pipeline_config, device=device, use_hpip=use_hpip, hpi_params=hpi_params
    )
    app = create_pipeline_app(pipeline, pipeline_config)
    run_server(app, host=host, port=port, debug=debug)


# for CLI
def main():
    """API for commad line"""
    args = args_cfg()
    if args.cmd is None:
        if args.install:
            install(args)
        else:
            pipeline_predict(
                args.pipeline,
                args.input,
                args.device,
                args.save_dir,
                use_hpip=args.use_hpip,
                serial_number=args.serial_number,
                update_license=args.update_license,
            )
    elif args.cmd == "serve":
        serve(
            args.pipeline,
            device=args.device,
            use_hpip=args.use_hpip,
            serial_number=args.serial_number,
            update_license=args.update_license,
            host=args.host,
            port=args.port,
            debug=args.debug,
        )
    else:
        raise AssertionError(f"Unknown command {args.cmd}")
