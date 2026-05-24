#!/usr/bin/env bash
# Reads .env and writes WebDAV creds into .dvc/config.local (gitignored).
# Required once per fresh clone before `dvc push` / `dvc pull` on Yandex.Disk.
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -f .env ]]; then
    echo "Missing .env (copy .env.example and fill in your Yandex creds)" >&2
    exit 1
fi

set -a
# shellcheck disable=SC1091
source .env
set +a

: "${WEBDAV_USER:?WEBDAV_USER must be set in .env}"
: "${WEBDAV_PASSWORD:?WEBDAV_PASSWORD must be set in .env}"

uv run dvc remote modify         data   user     "$WEBDAV_USER"
uv run dvc remote modify --local data   password "$WEBDAV_PASSWORD"
uv run dvc remote modify         models user     "$WEBDAV_USER"
uv run dvc remote modify --local models password "$WEBDAV_PASSWORD"

echo "DVC WebDAV auth configured for user '$WEBDAV_USER'."
