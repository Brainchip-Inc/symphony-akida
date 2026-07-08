#!/bin/bash
# SOAM service-instance wrapper. PEM invokes this with the ServiceContainer path
# as $1. Symphony's soamapi binding is Python 3.6-only, so the ServiceContainer
# runs under python3.6 and spawns the python3.12 akida worker itself.
#
# SOAM launches the wrapper with an EMPTY PATH, so set it explicitly first.
export PATH="/usr/bin:/usr/local/bin:/bin:/usr/sbin:/sbin"

EGO_TOP=/opt/ibm/spectrumcomputing
SOAM_LIB="$EGO_TOP/soam/7.3.2/linux-x86_64/lib64"
PY36=/usr/bin/python3.6

source "$EGO_TOP/profile.platform" >/dev/null 2>&1 || true

# soamapi 3.6 binding + akida native libs on the loader path.
export LD_LIBRARY_PATH="$SOAM_LIB:$EGO_TOP/soam/7.3.2/linux-x86_64/lib:/opt/akida-venv/lib/python3.12/site-packages/akida.libs:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$SOAM_LIB/pythonapi_3.6.7:${PYTHONPATH:-}"

# Worker location + interpreter (the worker is deployed next to $1).
DEPLOY_DIR="$(dirname "$1")"
export AKIDA_WORKER_PY="$DEPLOY_DIR/akida_worker.py"
export AKIDA_PYTHON="/opt/python3.12/bin/python3.12"
export AKIDA_VENV_SITEPACKAGES="/opt/akida-venv/lib/python3.12/site-packages"

# Per-node settings (AKIDA_DEVICE_INDEX etc.) written by the entrypoint from the
# container env; sourced here because SOAM does not inherit the container env.
[ -f /opt/akida-service/node.env ] && source /opt/akida-service/node.env

LOGDIR="/shared/soam/akida-service/logs"
mkdir -p "$LOGDIR"
exec >>"$LOGDIR/si-$(hostname -s)-$$.log" 2>&1

echo "[wrapper] $(date -u) starting on $(hostname -f)  device_index=${AKIDA_DEVICE_INDEX:-0}"
exec "$PY36" "$@"
