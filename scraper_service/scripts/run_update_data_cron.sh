#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)

JOB_NAME=update_data
LOG_DIR=${CRON_LOG_DIR:-"$ROOT_DIR/scraper_service/logs/cron"}
LOG_FILE="$LOG_DIR/$JOB_NAME.log"
LOCK_FILE="$LOG_DIR/$JOB_NAME.lock"
STATE_FILE="$LOG_DIR/$JOB_NAME.state"
LAST_START_FILE="$LOG_DIR/$JOB_NAME.last_start"
LAST_END_FILE="$LOG_DIR/$JOB_NAME.last_end"
LAST_SUCCESS_FILE="$LOG_DIR/$JOB_NAME.last_success"
LAST_EXIT_FILE="$LOG_DIR/$JOB_NAME.last_exit"

IMAGE_NAME=${IMAGE_NAME:-scraper-service}
HEADLESS=${HEADLESS:-1}
DOCKER_RUNTIME_VOLUME=${DOCKER_RUNTIME_VOLUME:-scraper-service-runtime}
PLAYWRIGHT_USER_DATA_DIR_IN_CONTAINER=${PLAYWRIGHT_USER_DATA_DIR_IN_CONTAINER:-/runtime/playwright-user-data}
SCRAPER_STORAGE_DIR_IN_CONTAINER=${SCRAPER_STORAGE_DIR_IN_CONTAINER:-/runtime/storage}
CRON_BUILD=${CRON_BUILD:-0}

timestamp() {
  date '+%Y-%m-%d %H:%M:%S %z'
}

write_marker() {
  printf '%s\n' "$2" > "$1"
}

append_log() {
  printf '%s | %s\n' "$(timestamp)" "$1" >> "$LOG_FILE"
}

ensure_image() {
  if [[ "$CRON_BUILD" == "1" ]] || ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
    append_log "Construyendo imagen Docker $IMAGE_NAME"
    docker build -f scraper_service/Dockerfile -t "$IMAGE_NAME" scraper_service >>"$LOG_FILE" 2>&1
  else
    append_log "Usando imagen Docker existente $IMAGE_NAME"
  fi
}

run_job() {
  ensure_image
  docker volume create "$DOCKER_RUNTIME_VOLUME" >/dev/null
  docker run --rm \
    -e HEADLESS="$HEADLESS" \
    -e PLAYWRIGHT_USER_DATA_DIR="$PLAYWRIGHT_USER_DATA_DIR_IN_CONTAINER" \
    -e SCRAPER_STORAGE_DIR="$SCRAPER_STORAGE_DIR_IN_CONTAINER" \
    -v "$ROOT_DIR:/app" \
    -v "$DOCKER_RUNTIME_VOLUME:/runtime" \
    -w /app \
    "$IMAGE_NAME" \
    python -m scraper_service.update_data
}

mkdir -p "$LOG_DIR"
touch "$LOG_FILE"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  write_marker "$STATE_FILE" "locked"
  write_marker "$LAST_END_FILE" "$(timestamp)"
  write_marker "$LAST_EXIT_FILE" "99"
  append_log "Otro proceso de $JOB_NAME sigue activo. Se omite esta corrida."
  exit 99
fi

start_ts=$(timestamp)
write_marker "$STATE_FILE" "running"
write_marker "$LAST_START_FILE" "$start_ts"
append_log "Iniciando job $JOB_NAME"

set +e
run_job >>"$LOG_FILE" 2>&1
exit_code=$?
set -e

end_ts=$(timestamp)
write_marker "$LAST_END_FILE" "$end_ts"
write_marker "$LAST_EXIT_FILE" "$exit_code"

if [[ "$exit_code" -eq 0 ]]; then
  write_marker "$STATE_FILE" "success"
  write_marker "$LAST_SUCCESS_FILE" "$end_ts"
  append_log "Job $JOB_NAME finalizado correctamente."
else
  write_marker "$STATE_FILE" "error"
  append_log "Job $JOB_NAME finalizado con error. exit_code=$exit_code"
fi

exit "$exit_code"
