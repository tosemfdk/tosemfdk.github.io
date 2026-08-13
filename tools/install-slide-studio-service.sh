#!/usr/bin/env bash

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STUDIO_DIR="$ROOT_DIR/slide-studio"
LABEL="com.tosemfdk.slide-studio"
DOMAIN="gui/$(id -u)"
TARGET_PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
TOKEN_DIR="$HOME/.config/tosemfdk"
TOKEN_FILE="$TOKEN_DIR/slide-studio-token"
DATA_DIR="${SLIDE_STUDIO_DATA_DIR:-$HOME/.local/share/tosemfdk-slide-studio}"
PUBLIC_URL="${SLIDE_STUDIO_PUBLIC_URL:-https://slides.tosemfdk.com}"
CODEX_BIN="${SLIDE_STUDIO_CODEX_BINARY:-$(command -v codex || true)}"

for executable in /opt/homebrew/bin/node /opt/homebrew/bin/npm; do
  if [[ ! -x "$executable" ]]; then
    echo "Required executable is missing: $executable" >&2
    exit 1
  fi
done
if [[ -z "$CODEX_BIN" || ! -x "$CODEX_BIN" ]]; then
  echo "Codex CLI is missing. Install it and complete 'codex login' before installing Slide Studio." >&2
  exit 1
fi

mkdir -p "$HOME/Library/LaunchAgents" "$HOME/Library/Logs/tosemfdk" "$TOKEN_DIR" "$DATA_DIR"
chmod 700 "$TOKEN_DIR"
if [[ ! -s "$TOKEN_FILE" ]]; then
  umask 077
  openssl rand -hex 32 > "$TOKEN_FILE"
fi
chmod 600 "$TOKEN_FILE"

cd "$STUDIO_DIR"
/opt/homebrew/bin/npm ci
/opt/homebrew/bin/npm run build

cat > "$TARGET_PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array><string>/bin/bash</string><string>$STUDIO_DIR/tools/run-production.sh</string></array>
  <key>WorkingDirectory</key><string>$STUDIO_DIR</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>SLIDE_STUDIO_ADMIN_TOKEN_FILE</key><string>$TOKEN_FILE</string>
    <key>SLIDE_STUDIO_DATA_DIR</key><string>$DATA_DIR</string>
    <key>SLIDE_STUDIO_PUBLIC_URL</key><string>$PUBLIC_URL</string>
    <key>SLIDE_STUDIO_HOST</key><string>127.0.0.1</string>
    <key>SLIDE_STUDIO_PORT</key><string>5560</string>
    <key>SLIDE_STUDIO_CODEX_BINARY</key><string>$CODEX_BIN</string>
  </dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><dict><key>SuccessfulExit</key><false/></dict>
  <key>ProcessType</key><string>Background</string>
  <key>ThrottleInterval</key><integer>5</integer>
  <key>StandardOutPath</key><string>$HOME/Library/Logs/tosemfdk/slide-studio.log</string>
  <key>StandardErrorPath</key><string>$HOME/Library/Logs/tosemfdk/slide-studio-error.log</string>
</dict>
</plist>
PLIST

if [[ -n "${SLIDE_STUDIO_ADMIN_EMAIL:-}" ]]; then
  /usr/libexec/PlistBuddy -c "Add :EnvironmentVariables:SLIDE_STUDIO_ADMIN_EMAIL string $SLIDE_STUDIO_ADMIN_EMAIL" "$TARGET_PLIST"
fi

plutil -lint "$TARGET_PLIST"
launchctl bootout "$DOMAIN/$LABEL" >/dev/null 2>&1 || true
launchctl bootstrap "$DOMAIN" "$TARGET_PLIST"
launchctl enable "$DOMAIN/$LABEL"
launchctl kickstart -k "$DOMAIN/$LABEL"

if command -v pbcopy >/dev/null 2>&1; then
  pbcopy < "$TOKEN_FILE"
  echo "Slide Studio administrator token copied to the clipboard."
fi

echo "Installed and started $LABEL"
echo "Local health: http://127.0.0.1:5560/api/health"
echo "Data directory: $DATA_DIR"
