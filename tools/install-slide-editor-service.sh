#!/usr/bin/env bash

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL="com.tosemfdk.slide-editor"
DOMAIN="gui/$(id -u)"
SOURCE_PLIST="$ROOT_DIR/ops/$LABEL.plist"
TARGET_PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
TOKEN_DIR="$HOME/.config/tosemfdk"
TOKEN_FILE="$TOKEN_DIR/slide-editor-token"
NODE_BIN="/opt/homebrew/bin/node"

if [[ ! -x "$NODE_BIN" ]]; then
  echo "Node.js is missing at $NODE_BIN" >&2
  exit 1
fi

mkdir -p "$HOME/Library/LaunchAgents" "$HOME/Library/Logs/tosemfdk" "$TOKEN_DIR"
chmod 700 "$TOKEN_DIR"

if [[ ! -s "$TOKEN_FILE" ]]; then
  umask 077
  openssl rand -hex 32 > "$TOKEN_FILE"
fi
chmod 600 "$TOKEN_FILE"

mkdir -p "$ROOT_DIR/.slide-editor/drafts"
cp "$SOURCE_PLIST" "$TARGET_PLIST"
plutil -lint "$TARGET_PLIST"

launchctl bootout "$DOMAIN/$LABEL" >/dev/null 2>&1 || true
launchctl bootstrap "$DOMAIN" "$TARGET_PLIST"
launchctl enable "$DOMAIN/$LABEL"
launchctl kickstart -k "$DOMAIN/$LABEL"

if command -v pbcopy >/dev/null 2>&1; then
  pbcopy < "$TOKEN_FILE"
  echo "Editor key copied to the clipboard."
fi

echo "Installed and started $LABEL"
echo "Editor key file: $TOKEN_FILE"
