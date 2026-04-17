#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)

IMAGE_NAME=${IMAGE_NAME:-predictor-pipelines}

cd "$ROOT_DIR"

echo "Construyendo imagen $IMAGE_NAME"
exec docker build -f ml_core/docker/Dockerfile -t "$IMAGE_NAME" .
