#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)

IMAGE_NAME=${IMAGE_NAME:-scraper-service}
HEADLESS=${HEADLESS:-1}
DOCKER_RUNTIME_VOLUME=${DOCKER_RUNTIME_VOLUME:-scraper-service-runtime}
PLAYWRIGHT_USER_DATA_DIR_IN_CONTAINER=${PLAYWRIGHT_USER_DATA_DIR_IN_CONTAINER:-/runtime/playwright-user-data}
SCRAPER_STORAGE_DIR_IN_CONTAINER=${SCRAPER_STORAGE_DIR_IN_CONTAINER:-/runtime/storage}
DEFAULT_ARGENPROP_URL=${DEFAULT_ARGENPROP_URL:-http://argenprop.com/departamentos/venta/capital-federal?orden-masnuevos}

if [ "$#" -ge 1 ]; then
  TARGET_URL=$1
  shift
else
  TARGET_URL=$DEFAULT_ARGENPROP_URL
  echo "No se recibio URL. Usando default: $TARGET_URL"
fi

cd "$ROOT_DIR"

echo "Construyendo imagen $IMAGE_NAME"
docker build -f scraper_service/Dockerfile -t "$IMAGE_NAME" scraper_service

docker volume create "$DOCKER_RUNTIME_VOLUME" >/dev/null

echo "Ejecutando scraper_service.sync_daily_new_listings en Docker con el repo montado en /app"
exec docker run --rm -it \
  -e HEADLESS="$HEADLESS" \
  -e PLAYWRIGHT_USER_DATA_DIR="$PLAYWRIGHT_USER_DATA_DIR_IN_CONTAINER" \
  -e SCRAPER_STORAGE_DIR="$SCRAPER_STORAGE_DIR_IN_CONTAINER" \
  -v "$ROOT_DIR:/app" \
  -v "$DOCKER_RUNTIME_VOLUME:/runtime" \
  -w /app \
  "$IMAGE_NAME" \
  python -m scraper_service.sync_daily_new_listings "$TARGET_URL" "$@"
