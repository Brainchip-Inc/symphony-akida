#!/usr/bin/env bash
# Run scripts/verify_reference.py inside the demo image on one Akida chip.
#
#   scripts/verify_reference.sh                                  # 500 frames of the quickest kit
#   scripts/verify_reference.sh --frames all                     # all of it
#   scripts/verify_reference.sh --npz data/voc/voc2007_test_r448.npz --frames all
#
# A throwaway privileged container with exactly one chip exposed (the probe_chips.sh pattern),
# the repo's models/ and scripts/ mounted read-only, and the test kit's directory mounted at
# /data. Needs no cluster: this is the pre-flight check that the port itself is correct.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-symphony-akida}"
KITS_DIR="${AKIDA_KITS_DIR:-$HERE/data/voc}"
NPZ="${AKIDA_TEST_NPZ:-}"
CHIP="${AKIDA_CHIP_NODE:-}"

args=()
while [ $# -gt 0 ]; do
    case "$1" in
        --npz) NPZ="$2"; shift 2;;
        --chip) CHIP="$2"; shift 2;;
        *) args+=("$1"); shift;;
    esac
done

if [ -z "$NPZ" ]; then
    # Smallest kit in data/voc, so the default check is the quick one. -L to size the symlink
    # target rather than the link, since kits are symlinked in.
    NPZ=$( { for f in "$KITS_DIR"/*.npz; do [ -f "$f" ] && stat -Lc '%s %n' "$f"; done; } \
           | sort -n | head -1 | cut -d' ' -f2- ) || true
fi
[ -n "$NPZ" ] && [ -f "$NPZ" ] \
    || { echo "No test kit in $KITS_DIR (see data/voc/README.md); pass --npz <path>." >&2; exit 1; }
# Resolve before mounting: kits in data/voc are symlinks, and a bind mount of that directory
# would carry the link into the container without its target.
NPZ="$(realpath "$NPZ")"
[ -f "$HERE/models/tiled_yolov2_voc.fbz" ] || { echo "No model; run 'git lfs pull'." >&2; exit 1; }

if [ -z "$CHIP" ]; then
    CHIP=$(ls -d /dev/akd1500_* /dev/akida[0-9]* 2>/dev/null | head -1 | xargs -r basename)
fi
[ -n "$CHIP" ] || { echo "No Akida device found (/dev/akd1500_* or /dev/akida*)." >&2; exit 1; }
echo "[verify] chip=$CHIP  npz=$(basename "$NPZ")"

exec docker run --rm --privileged --device="/dev/$CHIP" \
    -v "$HERE/models:/models:ro" \
    -v "$HERE/scripts:/scripts:ro" \
    -v "$HERE/src/common:/opt/akida-common:ro" \
    -v "$(cd "$(dirname "$NPZ")" && pwd):/data:ro" \
    -e PYTHONPATH=/opt/akida-venv/lib/python3.12/site-packages \
    -e LD_LIBRARY_PATH=/opt/akida-venv/lib/python3.12/site-packages/akida.libs \
    -e AKIDA_COMMON_DIR=/opt/akida-common \
    --entrypoint /opt/python3.12/bin/python3.12 \
    "$IMAGE" /scripts/verify_reference.py --npz "/data/$(basename "$NPZ")" ${args[@]+"${args[@]}"}
