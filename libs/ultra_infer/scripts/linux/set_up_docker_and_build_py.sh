#!/bin/bash

# input
CONTAINER_NAME="${CONTAINER_NAME:-build_fd}"
WITH_GPU="${WITH_GPU:-ON}"
ENABLE_BENCHMARK="${ENABLE_BENCHMARK:-OFF}"
DEBUG="${DEBUG:-OFF}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10.0}"

DOCKER_IMAGE="ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle_manylinux_devel:cuda11.8-cudnn8.6-trt8.5-gcc8.2"

if [[ -z "$PADDLEINFERENCE_URL" ]]; then
    echo "Error: PADDLEINFERENCE_URL is not set."
    exit 1
fi

if [[ -z "$PADDLEINFERENCE_VERSION" ]]; then
    echo "Error: PADDLEINFERENCE_VERSION is not set."
    exit 1
fi

# Set variables
CMAKE_CXX_COMPILER="/usr/local/gcc-8.2/bin/g++"

# Get the current script directory and compute the directory to mount
SCRIPT_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")")"
ULTRAINFER_DIR="$(realpath "$SCRIPT_DIR/../../../")"

# Set the Docker startup command
if [ "$WITH_GPU" = "ON" ]; then
    DOCKER_CMD=$(cat << EOF
docker run --gpus all -it --name="${CONTAINER_NAME}" --shm-size=128g --net=host \
-v "${ULTRAINFER_DIR}":/workspace \
-e CMAKE_CXX_COMPILER="${CMAKE_CXX_COMPILER}" \
-e "http_proxy=${http_proxy}" \
-e "https_proxy=${https_proxy}" \
"${DOCKER_IMAGE}" /bin/bash -c "
cd /workspace && \
ldconfig && \
./ultra_infer/scripts/linux/_build_py.sh --with-gpu "${WITH_GPU}" --enable-benchmark "${ENABLE_BENCHMARK}" --python "${PYTHON_VERSION}" --paddleinference-url "${PADDLEINFERENCE_URL}" --paddleinference-version "${PADDLEINFERENCE_VERSION}" && \
tail -f /dev/null"
EOF
)
else
    DOCKER_CMD=$(cat << EOF
docker run -it --name="${CONTAINER_NAME}" --shm-size=128g --net=host \
-v "${ULTRAINFER_DIR}":/workspace \
-e CMAKE_CXX_COMPILER="${CMAKE_CXX_COMPILER}" \
-e "http_proxy=${http_proxy}" \
-e "https_proxy=${https_proxy}" \
"${DOCKER_IMAGE}" /bin/bash -c "
cd /workspace && \
./ultra_infer/scripts/linux/_build_py.sh --with-gpu "${WITH_GPU}" --enable-benchmark "${ENABLE_BENCHMARK}" --python "${PYTHON_VERSION}" --paddleinference-url "${PADDLEINFERENCE_URL}" --paddleinference-version "${PADDLEINFERENCE_VERSION}" && \
tail -f /dev/null"
EOF
)
fi

# If in debug mode, replace --rm with -it and keep the container running
if [ "$DEBUG" = "OFF" ]; then
    DOCKER_CMD="${DOCKER_CMD/-it/--rm}"
    DOCKER_CMD="${DOCKER_CMD/ && tail -f \/dev\/null/}"
fi

# Check if a Docker container with the same name already exists
if docker ps -a --format '{{.Names}}' | grep -Eq "^${CONTAINER_NAME}\$"; then
    echo "Error: A Docker container with the name '${CONTAINER_NAME}' already exists."
    echo "Please remove the existing container or choose a different container name."
    exit 1
fi

echo "Starting Docker container..."
eval "$DOCKER_CMD"
