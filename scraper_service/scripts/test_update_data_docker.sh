#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)

IMAGE_NAME=${IMAGE_NAME:-scraper-service}
HEADLESS=${HEADLESS:-1}

cd "$ROOT_DIR"

echo "Construyendo imagen $IMAGE_NAME"
docker build -f scraper_service/Dockerfile -t "$IMAGE_NAME" .

echo "Probando scraper_service.update_data en Docker con HEADLESS=$HEADLESS"
exec docker run --rm -it \
  -e HEADLESS="$HEADLESS" \
  -v "$ROOT_DIR:/app" \
  -w /app \
  "$IMAGE_NAME" \
  python -m scraper_service.update_data_test
