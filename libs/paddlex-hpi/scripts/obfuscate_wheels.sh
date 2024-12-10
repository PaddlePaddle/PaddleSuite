#!/usr/bin/env bash

for python_api_tag in cp38-cp38 cp39-cp39 cp310-cp310; do
    for wheel_path in wheels/original/paddlex_hpi-*.whl; do
        docker run \
            -it \
            -v "$(pwd)":/workspace \
            -e PYTHON_ABI_TAG="${python_api_tag}" \
            --rm \
            --user "$(id -u)":"$(id -g)" \
            pywhlobf/pywhlobf:22.1.0-cython3-manylinux2014_x86_64 \
            "/workspace/${wheel_path}" \
            /workspace/wheels/obfuscated
    done
done
