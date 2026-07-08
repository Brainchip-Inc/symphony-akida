#!/bin/bash
# Build the SOAM deploy package for AkidaGenericService: the wrapper +
# ServiceContainer + worker, tarred with no leading directory (SOAM extracts
# them straight into the deploy dir). The XML profile is registered separately.
#
#   Output: ${DEST:-<this dir>}/AkidaGenericServicePackage.v1.tar.gz
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DEST="${DEST:-$HERE}"
OUT="$DEST/AkidaGenericServicePackage.v1.tar.gz"

tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
cp "$HERE/run_akida_service.sh" "$HERE/AkidaServiceContainer.py" "$HERE/akida_worker.py" "$tmp/"
chmod +x "$tmp/run_akida_service.sh"
tar -C "$tmp" -czf "$OUT" run_akida_service.sh AkidaServiceContainer.py akida_worker.py
echo "[ok] built $OUT"
tar -tzf "$OUT" | sed 's/^/    /'
