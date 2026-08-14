#!/usr/bin/env bash
# Launch the serial-http-round-robin dashboard on the laptop (host tooling via uv).
# Sizes the node list to the Akida chip count and serves http://localhost:5001.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"

# Count Akida chip nodes of either family (AKD1500 /dev/akd1500_<N> + AKD1000 /dev/akida<N>),
# capped at the CE 64-core limit (master + 7 compute) so it matches what scripts/launch/up.sh brings up.
_n=$(ls -d /dev/akd1500_* /dev/akida[0-9]* 2>/dev/null | grep -Ec 'akd1500_[0-9]+|akida[0-9]+' || true)
[ "${_n:-0}" -ge 1 ] || _n=3
[ "$_n" -gt 7 ] && _n=7
export AKIDA_NODE_COUNT="${AKIDA_NODE_COUNT:-$_n}"
export FLASK_PORT="${FLASK_PORT:-5001}"

cd "$REPO"
exec uv run python src/apps/serial-http-round-robin/dashboard/app.py
