#!/usr/bin/env bash

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL="com.tosemfdk.web"
DOMAIN="gui/$(id -u)"
SOURCE_PLIST="$ROOT_DIR/ops/$LABEL.plist"
TARGET_PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
CADDY_BIN="/opt/homebrew/opt/caddy/bin/caddy"

if [[ ! -x "$CADDY_BIN" ]]; then
  echo "Caddy is missing. Install it with: brew install caddy" >&2
  exit 1
fi

mkdir -p "$HOME/Library/LaunchAgents" "$HOME/Library/Logs/tosemfdk"
"$CADDY_BIN" validate --config "$ROOT_DIR/ops/Caddyfile" --adapter caddyfile

if [[ ! -s "$ROOT_DIR/.deploy/current/index.html" ]]; then
  "$ROOT_DIR/tools/deploy.sh"
fi

launchctl bootout "$DOMAIN/$LABEL" >/dev/null 2>&1 || true
cp "$SOURCE_PLIST" "$TARGET_PLIST"
plutil -lint "$TARGET_PLIST"
launchctl bootstrap "$DOMAIN" "$TARGET_PLIST"
launchctl enable "$DOMAIN/$LABEL"
launchctl kickstart -k "$DOMAIN/$LABEL"

echo "Installed and started $LABEL"
