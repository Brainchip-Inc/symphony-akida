#!/bin/bash
# Per-node HTTP inference server wrapper (serial-http-round-robin app).
#
# The entrypoint launches this on each COMPUTE node when START_HTTP=1. It runs the
# akida 3.12 HTTP server (http_server.py) which maps a model hw_only on this node's
# chip and serves the /health /models /load /reload /unload /infer API the dashboard
# round-robins across. No SOAM here -- this is the plain-HTTP "before" path.
export PATH="/usr/bin:/usr/local/bin:/bin:/usr/sbin:/sbin"

AKIDA_PYTHON="/opt/python3.12/bin/python3.12"
VENV_SP="/opt/akida-venv/lib/python3.12/site-packages"
COMMON="/opt/akida-common"
HTTP_DIR="$(cd "$(dirname "$0")" && pwd)"

# akida native libs on the loader path; venv site-packages + shared core on PYTHONPATH.
export LD_LIBRARY_PATH="$VENV_SP/akida.libs:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$VENV_SP:$COMMON:${PYTHONPATH:-}"
export AKIDA_COMMON_DIR="$COMMON"

# Per-node settings (AKIDA_DEVICE_INDEX) written by the entrypoint from the container env.
[ -f /opt/akida-service/node.env ] && source /opt/akida-service/node.env
export AKIDA_DEFAULT_MODEL="${AKIDA_DEFAULT_MODEL:-kws_keyword_spotting_sparse}"
export AKIDA_MODELS_DIR="${AKIDA_MODELS_DIR:-/shared/models}"
export HTTP_PORT="${HTTP_PORT:-8790}"

LOGDIR="/shared/soam/http-service/logs"
mkdir -p "$LOGDIR"
exec >>"$LOGDIR/http-$(hostname -s)-$$.log" 2>&1

echo "[http-wrapper] $(date -u) starting on $(hostname -f) device_index=${AKIDA_DEVICE_INDEX:-0} port=$HTTP_PORT default=$AKIDA_DEFAULT_MODEL"
exec "$AKIDA_PYTHON" "$HTTP_DIR/http_server.py"
