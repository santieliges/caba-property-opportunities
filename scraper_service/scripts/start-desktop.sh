#!/usr/bin/env sh
set -eu

DISPLAY=${DISPLAY:-:99}
SCREEN_WIDTH=${SCREEN_WIDTH:-1440}
SCREEN_HEIGHT=${SCREEN_HEIGHT:-900}
SCREEN_DEPTH=${SCREEN_DEPTH:-24}
VNC_PORT=${VNC_PORT:-5900}
NOVNC_PORT=${NOVNC_PORT:-6080}
export DISPLAY

Xvfb "$DISPLAY" \
  -screen 0 "${SCREEN_WIDTH}x${SCREEN_HEIGHT}x${SCREEN_DEPTH}" \
  -ac \
  +extension RANDR \
  >/tmp/xvfb.log 2>&1 &

ready=false
attempt=0
while [ "$attempt" -lt 50 ]; do
  if xdpyinfo -display "$DISPLAY" >/dev/null 2>&1; then
    ready=true
    break
  fi
  attempt=$((attempt + 1))
  sleep 0.1
done

if [ "$ready" != true ]; then
  echo "Xvfb did not become ready on DISPLAY=$DISPLAY" >&2
  cat /tmp/xvfb.log >&2
  exit 1
fi

fluxbox >/tmp/fluxbox.log 2>&1 &

if [ -n "${NOVNC_PASSWORD:-}" ]; then
  password_file=/tmp/x11vnc.pass
  x11vnc -storepasswd "$NOVNC_PASSWORD" "$password_file" >/dev/null
  vnc_auth="-rfbauth $password_file"
else
  echo "Warning: noVNC is running without authentication." >&2
  vnc_auth=-nopw
fi

# x11vnc only accepts local connections; websockify is the public endpoint.
# shellcheck disable=SC2086
x11vnc \
  -display "$DISPLAY" \
  -forever \
  -shared \
  -localhost \
  -rfbport "$VNC_PORT" \
  $vnc_auth \
  >/tmp/x11vnc.log 2>&1 &

websockify \
  --web=/usr/share/novnc \
  "$NOVNC_PORT" \
  "localhost:$VNC_PORT" \
  >/tmp/novnc.log 2>&1 &

echo "noVNC available on port $NOVNC_PORT (DISPLAY=$DISPLAY)"
exec "$@"
