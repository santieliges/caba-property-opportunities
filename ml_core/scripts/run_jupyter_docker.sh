#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)

IMAGE_NAME=${IMAGE_NAME:-predictor-pipelines}
PORT=${PORT:-8888}

cd "$ROOT_DIR"

echo "Construyendo imagen $IMAGE_NAME"
docker build -f ml_core/docker/Dockerfile -t "$IMAGE_NAME" .

echo "Levantando Jupyter Lab en http://localhost:$PORT"
exec docker run --rm -it \
  -p "$PORT:8888" \
  -v "$ROOT_DIR:/workspace" \
  -w /workspace \
  "$IMAGE_NAME" \
  jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token=
