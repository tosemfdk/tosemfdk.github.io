#!/usr/bin/env bash

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPLOY_ROOT="$ROOT_DIR/.deploy"
RELEASES_DIR="$DEPLOY_ROOT/releases"
CURRENT_LINK="$DEPLOY_ROOT/current"

if [[ ! -L "$CURRENT_LINK" ]]; then
  echo "No active release exists." >&2
  exit 1
fi

current_release="$(basename "$(readlink "$CURRENT_LINK")")"
previous_release="$(
  find "$RELEASES_DIR" -mindepth 1 -maxdepth 1 -type d -print \
    | LC_ALL=C sort -r \
    | while IFS= read -r release; do
        [[ "$(basename "$release")" == "$current_release" ]] || {
          printf '%s\n' "$release"
          break
        }
      done
)"

if [[ -z "$previous_release" ]]; then
  echo "No previous release is available." >&2
  exit 1
fi

new_link="$DEPLOY_ROOT/.rollback-$$"
trap 'rm -f "$new_link"' EXIT
ln -s "releases/$(basename "$previous_release")" "$new_link"
mv -fh "$new_link" "$CURRENT_LINK"
trap - EXIT

echo "Rolled back from $current_release to $(basename "$previous_release")"
