#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)

PYTHON_BIN=${PYTHON_BIN:-"$ROOT_DIR/venv/bin/python"}
HEADLESS=${HEADLESS:-1}

if [ ! -x "$PYTHON_BIN" ]; then
  echo "No se encontro un interprete ejecutable en: $PYTHON_BIN" >&2
  exit 1
fi

cd "$ROOT_DIR"
export HEADLESS

echo "Probando scraper_service.scrap_new_data con HEADLESS=$HEADLESS"
exec "$PYTHON_BIN" -m scraper_service.scrap_new_data_test
