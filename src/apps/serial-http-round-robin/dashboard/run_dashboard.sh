#!/usr/bin/env bash
# Launch the serial-http-round-robin dashboard on the laptop (host tooling via uv).
# Sizes the node list to the Akida chip count and serves http://localhost:5001.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"

export AKIDA_NODE_COUNT="${AKIDA_NODE_COUNT:-$(ls -d /dev/akida* 2>/dev/null | grep -Ec 'akida[0-9]+' || echo 3)}"
export FLASK_PORT="${FLASK_PORT:-5001}"

cd "$REPO"
exec uv run python src/apps/serial-http-round-robin/dashboard/app.py
