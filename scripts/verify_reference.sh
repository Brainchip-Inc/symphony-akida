#!/usr/bin/env bash
# Run scripts/verify_reference.py inside the demo image on one Akida chip.
#
#   scripts/verify_reference.sh                                  # the committed kit, 500 frames max
#   scripts/verify_reference.sh --frames all                     # all of it
#   scripts/verify_reference.sh --npz ~/data/voc/VOCdevkit/voc2007_test_r448.npz --frames all
#
# A throwaway privileged container with exactly one chip exposed (the probe_chips.sh pattern),
# the repo's models/ and scripts/ mounted read-only, and the test kit's directory mounted at
# /data. Needs no cluster: this is the pre-flight check that the port itself is correct.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-symphony-akida}"
KIT_DIR="${AKIDA_KIT_DIR:-$HERE/data/voc2007}"
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
    # A dataset folder holds exactly one .npz, so the committed kit is simply what is in there.
    NPZ=$( { for f in "$KIT_DIR"/*.npz; do [ -f "$f" ] && echo "$f"; done; } | head -1 ) || true
fi
[ -n "$NPZ" ] && [ -f "$NPZ" ] \
    || { echo "No test kit in $KIT_DIR (see data/README.md); pass --npz <path>." >&2; exit 1; }
# ~130 bytes of pointer text bind-mounts and opens like a file, then fails deep inside TestKit
# with a zip error. Catch it here, where the fix is one command.
if [ "$(stat -Lc %s "$NPZ")" -lt 1024 ] && read -r first < "$NPZ" 2>/dev/null \
   && [ "$first" = "version https://git-lfs.github.com/spec/v1" ]; then
    echo "$NPZ is a Git LFS pointer, not the kit. Run: git lfs install && git lfs pull" >&2
    exit 1
fi
# Resolve before mounting: a kit passed with --npz may be a symlink, and a bind mount of its
# directory would carry the link into the container without its target.
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
