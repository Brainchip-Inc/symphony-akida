#!/usr/bin/env bash
# Launch the serial-http-round-robin dashboard on the laptop (host tooling via uv).
# Serves http://localhost:5001.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"

# The dashboard discovers the nodes and their published ports from the running compute
# containers (src/common/fleet.py). Counting /dev nodes here, which is what this used to do,
# gets the number wrong on a mixed host and cannot see a node published off the usual port.
# AKIDA_NODE_COUNT and AKIDA_NODES still override, for a fleet docker cannot see.
export FLASK_PORT="${FLASK_PORT:-5001}"

cd "$REPO"
exec uv run python src/apps/serial-http-round-robin/dashboard/app.py
