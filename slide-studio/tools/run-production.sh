#!/usr/bin/env bash

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOKEN_FILE="${SLIDE_STUDIO_ADMIN_TOKEN_FILE:-$HOME/.config/tosemfdk/slide-studio-token}"

if [[ -z "${SLIDE_STUDIO_ADMIN_EMAIL:-}" ]]; then
  if [[ ! -s "$TOKEN_FILE" ]]; then
    echo "Slide Studio token file is missing: $TOKEN_FILE" >&2
    exit 1
  fi
  export SLIDE_STUDIO_ADMIN_TOKEN="$(<"$TOKEN_FILE")"
fi

export NODE_ENV=production
export SLIDE_STUDIO_HOST="${SLIDE_STUDIO_HOST:-127.0.0.1}"
export SLIDE_STUDIO_PORT="${SLIDE_STUDIO_PORT:-5560}"
export SLIDE_STUDIO_DATA_DIR="${SLIDE_STUDIO_DATA_DIR:-$HOME/.local/share/tosemfdk-slide-studio}"

cd "$ROOT_DIR"
exec /opt/homebrew/bin/node dist-server/index.js
