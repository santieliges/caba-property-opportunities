#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)

IMAGE_NAME=${IMAGE_NAME:-predictor-pipelines}
DATASET=${DATASET:-venta}

cd "$ROOT_DIR"

echo "Construyendo imagen $IMAGE_NAME"
docker build -f pipelines/docker/Dockerfile -t "$IMAGE_NAME" .

echo "Ejecutando preprocessing dentro del contenedor para dataset=$DATASET"
exec docker run --rm \
  -v "$ROOT_DIR:/workspace" \
  -w /workspace \
  "$IMAGE_NAME" \
  python -m pipelines.preprocessing.build_processed_data --dataset "$DATASET"
