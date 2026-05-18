#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)
LOG_DIR=${CRON_LOG_DIR:-"$ROOT_DIR/scraper_service/logs/cron"}

print_value() {
  local file_path=$1
  if [[ -f "$file_path" ]]; then
    cat "$file_path"
  else
    printf 'n/a\n'
  fi
}

show_job() {
  local job_name=$1
  local log_file="$LOG_DIR/$job_name.log"

  printf '[%s]\n' "$job_name"
  printf 'state=%s\n' "$(print_value "$LOG_DIR/$job_name.state")"
  printf 'last_start=%s\n' "$(print_value "$LOG_DIR/$job_name.last_start")"
  printf 'last_end=%s\n' "$(print_value "$LOG_DIR/$job_name.last_end")"
  printf 'last_success=%s\n' "$(print_value "$LOG_DIR/$job_name.last_success")"
  printf 'last_exit=%s\n' "$(print_value "$LOG_DIR/$job_name.last_exit")"
  printf 'log_file=%s\n' "$log_file"
  if [[ -f "$log_file" ]]; then
    printf 'log_mtime=%s\n' "$(date -r "$log_file" '+%Y-%m-%d %H:%M:%S %z')"
  else
    printf 'log_mtime=n/a\n'
  fi
  printf '\n'
}

show_job update_data
show_job sync_daily_new_listings
