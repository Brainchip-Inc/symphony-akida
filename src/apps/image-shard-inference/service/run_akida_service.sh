#!/bin/bash
# SOAM wrapper for the shard pipeline's INFERENCE service instance. PEM invokes it with the
# ServiceContainer path as $1. soamapi is python3.6-only, so the ServiceContainer runs under
# python3.6 and spawns the python3.12 akida worker (inference_worker.py) itself.
#
# SOAM launches the wrapper with an EMPTY PATH and env, so set everything explicitly.
export PATH="/usr/bin:/usr/local/bin:/bin:/usr/sbin:/sbin"

EGO_TOP=/opt/ibm/spectrumcomputing
SOAM_LIB="$EGO_TOP/soam/7.3.2/linux-x86_64/lib64"
PY36=/usr/bin/python3.6

source "$EGO_TOP/profile.platform" >/dev/null 2>&1 || true

# soamapi 3.6 binding + akida native libs on the loader path.
export LD_LIBRARY_PATH="$SOAM_LIB:$EGO_TOP/soam/7.3.2/linux-x86_64/lib:/opt/akida-venv/lib/python3.12/site-packages/akida.libs:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$SOAM_LIB/pythonapi_3.6.7:${PYTHONPATH:-}"

# The python3.12 akida worker is deployed next to the container ($1).
DEPLOY_DIR="$(dirname "$1")"
export AKIDA_WORKER_PY="$DEPLOY_DIR/inference_worker.py"
export AKIDA_PYTHON="/opt/python3.12/bin/python3.12"
export AKIDA_VENV_SITEPACKAGES="/opt/akida-venv/lib/python3.12/site-packages"

# Per-node settings (AKIDA_DEVICE_INDEX, AKIDA_SHM_BYTES) written by the entrypoint into this
# app's dir; sourced here because SOAM does not inherit the container env.
[ -f /opt/akida-shard-service/node.env ] && source /opt/akida-shard-service/node.env
export AKIDA_SHM_BYTES="${AKIDA_SHM_BYTES:-8388608}"
export AKIDA_PIPELINE_DIR="${AKIDA_PIPELINE_DIR:-/shared/pipeline}"

LOGDIR="/shared/soam/shard-inference/logs"
mkdir -p "$LOGDIR"
exec >>"$LOGDIR/si-$(hostname -s)-$$.log" 2>&1

echo "[wrapper] $(date -u) shard-inference on $(hostname -f)  device_index=${AKIDA_DEVICE_INDEX:-0}"
exec "$PY36" "$@"
