#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)

IMAGE_NAME=${IMAGE_NAME:-scraper-service}
HEADLESS=${HEADLESS:-0}
NOVNC_BIND_ADDRESS=${NOVNC_BIND_ADDRESS:-127.0.0.1}
NOVNC_PORT_HOST=${NOVNC_PORT_HOST:-6080}
NOVNC_PASSWORD=${NOVNC_PASSWORD:-}

cd "$ROOT_DIR"

echo "Construyendo imagen $IMAGE_NAME"
docker build -f scraper_service/Dockerfile -t "$IMAGE_NAME" scraper_service

echo "Ejecutando scraper_service.scrape_caba_bulk_listings en Docker con el repo montado en /app"
exec docker run --rm -it \
  -p "$NOVNC_BIND_ADDRESS:$NOVNC_PORT_HOST:6080" \
  -e NOVNC_PASSWORD="$NOVNC_PASSWORD" \
  -e HEADLESS="$HEADLESS" \
  -v "$ROOT_DIR:/app" \
  -w /app \
  "$IMAGE_NAME" \
  python -m scraper_service.scrape_caba_bulk_listings
