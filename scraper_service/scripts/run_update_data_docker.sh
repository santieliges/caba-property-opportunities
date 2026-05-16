#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)

IMAGE_NAME=${IMAGE_NAME:-scraper-service}
HEADLESS=${HEADLESS:-1}
DOCKER_RUNTIME_VOLUME=${DOCKER_RUNTIME_VOLUME:-scraper-service-runtime}
PLAYWRIGHT_USER_DATA_DIR_IN_CONTAINER=${PLAYWRIGHT_USER_DATA_DIR_IN_CONTAINER:-/runtime/playwright-user-data}
SCRAPER_STORAGE_DIR_IN_CONTAINER=${SCRAPER_STORAGE_DIR_IN_CONTAINER:-/runtime/storage}

cd "$ROOT_DIR"

echo "Construyendo imagen $IMAGE_NAME"
docker build -f scraper_service/Dockerfile -t "$IMAGE_NAME" scraper_service

docker volume create "$DOCKER_RUNTIME_VOLUME" >/dev/null

echo "Ejecutando scraper_service.update_data en Docker con el repo montado en /app"
exec docker run --rm -it \
  -e HEADLESS="$HEADLESS" \
  -e PLAYWRIGHT_USER_DATA_DIR="$PLAYWRIGHT_USER_DATA_DIR_IN_CONTAINER" \
  -e SCRAPER_STORAGE_DIR="$SCRAPER_STORAGE_DIR_IN_CONTAINER" \
  -v "$ROOT_DIR:/app" \
  -v "$DOCKER_RUNTIME_VOLUME:/runtime" \
  -w /app \
  "$IMAGE_NAME" \
  python -m scraper_service.update_data
