#!/usr/bin/env bash

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUBY_HOME="${RUBY_HOME:-/opt/homebrew/opt/ruby@3.4}"
DEPLOY_ROOT="$ROOT_DIR/.deploy"
RELEASES_DIR="$DEPLOY_ROOT/releases"
RELEASE_ID="$(date -u '+%Y%m%dT%H%M%SZ')-$$"
STAGING_DIR="$DEPLOY_ROOT/.staging-$RELEASE_ID"
RELEASE_DIR="$RELEASES_DIR/$RELEASE_ID"
CURRENT_LINK="$DEPLOY_ROOT/current"
NEW_LINK="$DEPLOY_ROOT/.current-$RELEASE_ID"

export PATH="$RUBY_HOME/bin:/opt/homebrew/lib/ruby/gems/3.4.0/bin:$PATH"
export BUNDLE_GEMFILE="$ROOT_DIR/Gemfile"

cleanup() {
  rm -rf "$STAGING_DIR"
  rm -f "$NEW_LINK"
}
trap cleanup EXIT

if [[ ! -x "$RUBY_HOME/bin/ruby" ]]; then
  echo "Ruby 3.4 is missing. Install it with: brew install ruby@3.4" >&2
  exit 1
fi

cd "$ROOT_DIR"
mkdir -p "$RELEASES_DIR"

bundle config set --local path vendor/bundle >/dev/null
if ! bundle check >/dev/null 2>&1; then
  bundle install
fi

echo "Building release $RELEASE_ID"
JEKYLL_ENV=production bundle exec jekyll build --destination "$STAGING_DIR"

echo "Checking generated links"
bundle exec htmlproofer "$STAGING_DIR" \
  --disable-external \
  --ignore-urls "/^http:\/\/127.0.0.1/,/^http:\/\/0.0.0.0/,/^http:\/\/localhost/"

for required_file in index.html sitemap.xml feed.xml; do
  if [[ ! -s "$STAGING_DIR/$required_file" ]]; then
    echo "Build is missing required file: $required_file" >&2
    exit 1
  fi
done

mv "$STAGING_DIR" "$RELEASE_DIR"
ln -s "releases/$RELEASE_ID" "$NEW_LINK"
mv -fh "$NEW_LINK" "$CURRENT_LINK"

find "$RELEASES_DIR" -mindepth 1 -maxdepth 1 -type d -print \
  | LC_ALL=C sort -r \
  | tail -n +4 \
  | while IFS= read -r old_release; do
      rm -rf "$old_release"
    done

trap - EXIT
echo "Deployed $RELEASE_ID"
