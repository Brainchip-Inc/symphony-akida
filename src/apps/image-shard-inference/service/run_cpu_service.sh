#!/bin/bash
# SOAM wrapper for the shard pipeline's CPU-only service instances (SEGMENT and STITCH). PEM
# invokes it with the ServiceContainer path as $1. These stages never touch a chip -- one cuts
# tiles, the other merges detections -- so they run on the management host. Both are numpy
# though, and soamapi is python3.6-only, so the container spawns a python3.12 worker just as
# the inference stage does; the akida native libs are deliberately NOT on the loader path here.
#
# SOAM launches the wrapper with an EMPTY PATH and env, so set everything explicitly.
export PATH="/usr/bin:/usr/local/bin:/bin:/usr/sbin:/sbin"

EGO_TOP=/opt/ibm/spectrumcomputing
SOAM_LIB="$EGO_TOP/soam/7.3.2/linux-x86_64/lib64"
PY36=/usr/bin/python3.6

source "$EGO_TOP/profile.platform" >/dev/null 2>&1 || true

export LD_LIBRARY_PATH="$SOAM_LIB:$EGO_TOP/soam/7.3.2/linux-x86_64/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$SOAM_LIB/pythonapi_3.6.7:${PYTHONPATH:-}"

# The python3.12 worker is deployed next to the container ($1), named after the stage.
DEPLOY_DIR="$(dirname "$1")"
STAGE="$(basename "$1" .py)"
case "$STAGE" in
    SegmentServiceContainer) WORKER=segment_worker.py ;;
    StitchServiceContainer)  WORKER=stitch_worker.py ;;
    *) echo "[wrapper] unknown stage $STAGE" >&2; exit 1 ;;
esac
export AKIDA_WORKER_PY="$DEPLOY_DIR/$WORKER"
export AKIDA_PYTHON="/opt/python3.12/bin/python3.12"
export AKIDA_VENV_SITEPACKAGES="/opt/akida-venv/lib/python3.12/site-packages"
export AKIDA_COMMON_DIR="${AKIDA_COMMON_DIR:-/opt/akida-common}"

export AKIDA_PIPELINE_DIR="${AKIDA_PIPELINE_DIR:-/shared/pipeline}"
export AKIDA_MODELS_DIR="${AKIDA_MODELS_DIR:-/shared/models}"
export AKIDA_SHM_BYTES="${AKIDA_SHM_BYTES:-8388608}"

LOGDIR="/shared/soam/shard-cpu/logs"
mkdir -p "$LOGDIR"
exec >>"$LOGDIR/${STAGE}-$(hostname -s)-$$.log" 2>&1

echo "[wrapper] $(date -u) $STAGE on $(hostname -f)  worker=$WORKER"
exec "$PY36" "$@"
