#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)

BEGIN_MARKER="# BEGIN caba-property-opportunities cron jobs"
END_MARKER="# END caba-property-opportunities cron jobs"

UPDATE_SCHEDULE=${UPDATE_SCHEDULE:-"0 0 * * *"}
SYNC_SCHEDULE=${SYNC_SCHEDULE:-"0 3 * * *"}
CRON_TZ_VALUE=${CRON_TZ_VALUE:-"America/Sao_Paulo"}

UPDATE_SCRIPT="$ROOT_DIR/scraper_service/scripts/run_update_data_cron.sh"
SYNC_SCRIPT="$ROOT_DIR/scraper_service/scripts/run_sync_daily_new_listings_cron.sh"
LEGACY_UPDATE_SCRIPT="$ROOT_DIR/scraper_service/scripts/run_update_data_docker.sh"
LEGACY_SYNC_SCRIPT="$ROOT_DIR/scraper_service/scripts/run_sync_daily_new_listings_docker.sh"

current_crontab=$(crontab -l 2>/dev/null || true)
filtered_crontab=$(
  printf '%s\n' "$current_crontab" | awk \
    -v begin="$BEGIN_MARKER" \
    -v end="$END_MARKER" \
    -v update_script="$UPDATE_SCRIPT" \
    -v sync_script="$SYNC_SCRIPT" \
    -v legacy_update_script="$LEGACY_UPDATE_SCRIPT" \
    -v legacy_sync_script="$LEGACY_SYNC_SCRIPT" '
    $0 == begin { skip = 1; next }
    $0 == end { skip = 0; next }
    index($0, update_script) > 0 { next }
    index($0, sync_script) > 0 { next }
    index($0, legacy_update_script) > 0 { next }
    index($0, legacy_sync_script) > 0 { next }
    !skip { print }
  '
)

new_crontab=$(cat <<EOF
$filtered_crontab
$BEGIN_MARKER
SHELL=/bin/bash
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
CRON_TZ=$CRON_TZ_VALUE
$UPDATE_SCHEDULE $UPDATE_SCRIPT
$SYNC_SCHEDULE $SYNC_SCRIPT
$END_MARKER
EOF
)

printf '%s\n' "$new_crontab" | crontab -
printf 'Cron instalado.\n'
crontab -l
