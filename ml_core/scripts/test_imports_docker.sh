#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)

IMAGE_NAME=${IMAGE_NAME:-predictor-pipelines}

cd "$ROOT_DIR"

echo "Construyendo imagen $IMAGE_NAME"
docker build -f pipelines/docker/Dockerfile -t "$IMAGE_NAME" .

echo "Probando imports de pipelines dentro del contenedor"
exec docker run --rm \
  -v "$ROOT_DIR:/workspace" \
  -w /workspace \
  "$IMAGE_NAME" \
  python - <<'PY'
from pipelines.preprocessing.preprocessing import (
    build_alquiler_processed_dataset,
    build_venta_processed_dataset,
)
from pipelines.models import (
    GWRModel,
    ModelEvaluator,
    RegressionKrigingModel,
    SpatialAutoregressiveModel,
    SpatialOutlierAnalyzer,
    SpatialOutlierDetector,
)

print("imports_ok")
print(GWRModel.__name__)
print(RegressionKrigingModel.__name__)
print(SpatialAutoregressiveModel.__name__)
print(ModelEvaluator.__name__)
print(SpatialOutlierDetector.__name__)
print(SpatialOutlierAnalyzer.__name__)
print(build_venta_processed_dataset.__name__)
print(build_alquiler_processed_dataset.__name__)
PY
