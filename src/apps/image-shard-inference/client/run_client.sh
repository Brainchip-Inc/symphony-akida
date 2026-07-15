#!/bin/bash
# Run the shard-pipeline SOAM client inside the master container. Sets up the Python 3.6 soamapi
# binding on the loader/module path, then runs shard_client.py.
#   docker exec symphony-master /opt/akida-shard-client/run_client.sh --count 200
EGO_TOP=/opt/ibm/spectrumcomputing
SOAM_LIB="$EGO_TOP/soam/7.3.2/linux-x86_64/lib64"
source "$EGO_TOP/profile.platform" >/dev/null 2>&1
export PYTHONPATH="$SOAM_LIB/pythonapi_3.6.7:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$SOAM_LIB:$EGO_TOP/soam/7.3.2/linux-x86_64/lib:${LD_LIBRARY_PATH:-}"
exec /usr/bin/python3.6 "$(dirname "$0")/shard_client.py" "$@"
