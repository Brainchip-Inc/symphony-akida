#!/bin/bash
# SOAM wrapper for the shard pipeline's CPU-only service instances (SEGMENT and STITCH). PEM
# invokes it with the ServiceContainer path as $1. These stages are pure python3.6 (no akida,
# no numpy, no worker subprocess) -- just byte-slicing / small-list math -- so they run on the
# management host and never touch a chip.
#
# SOAM launches the wrapper with an EMPTY PATH and env, so set everything explicitly.
export PATH="/usr/bin:/usr/local/bin:/bin:/usr/sbin:/sbin"

EGO_TOP=/opt/ibm/spectrumcomputing
SOAM_LIB="$EGO_TOP/soam/7.3.2/linux-x86_64/lib64"
PY36=/usr/bin/python3.6

source "$EGO_TOP/profile.platform" >/dev/null 2>&1 || true

export LD_LIBRARY_PATH="$SOAM_LIB:$EGO_TOP/soam/7.3.2/linux-x86_64/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$SOAM_LIB/pythonapi_3.6.7:${PYTHONPATH:-}"

export AKIDA_PIPELINE_DIR="${AKIDA_PIPELINE_DIR:-/shared/pipeline}"
export AKIDA_MODELS_DIR="${AKIDA_MODELS_DIR:-/shared/models}"

STAGE="$(basename "$1" .py)"
LOGDIR="/shared/soam/shard-cpu/logs"
mkdir -p "$LOGDIR"
exec >>"$LOGDIR/${STAGE}-$(hostname -s)-$$.log" 2>&1

echo "[wrapper] $(date -u) $STAGE on $(hostname -f)"
exec "$PY36" "$@"
