#!/bin/bash
# Build the three SOAM deploy packages for the shard pipeline (segment / inference / stitch).
# Each is tarred with NO leading directory (SOAM extracts straight into the deploy dir), so the
# containers can `import shard_common` / find inference_worker.py as co-located files.
#
#   Output: ${DEST:-<this dir>}/Shard{Segment,Inference,Stitch}ServicePackage.v1.tar.gz
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DEST="${DEST:-$HERE}"

pack() {  # <out-name> <file...>
    local out="$DEST/$1"; shift
    local tmp; tmp="$(mktemp -d)"
    cp "$@" "$tmp/"
    # make wrappers executable inside the package
    chmod +x "$tmp"/*.sh 2>/dev/null || true
    ( cd "$tmp" && tar -czf "$out" ./* )
    rm -rf "$tmp"
    echo "[ok] built $out"
    tar -tzf "$out" | sed 's/^/    /'
}

pack ShardSegmentServicePackage.v1.tar.gz \
    "$HERE/run_cpu_service.sh" "$HERE/shard_common.py" "$HERE/segment/SegmentServiceContainer.py"

pack ShardInferenceServicePackage.v1.tar.gz \
    "$HERE/run_akida_service.sh" "$HERE/inference/InferenceServiceContainer.py" \
    "$HERE/inference/inference_worker.py"

pack ShardStitchServicePackage.v1.tar.gz \
    "$HERE/run_cpu_service.sh" "$HERE/shard_common.py" "$HERE/stitch/StitchServiceContainer.py"
