#!/bin/bash
# Run the SOAM batch client inside the master container. Sets up the Python 3.6
# soamapi binding on the loader/module path, then runs soam_client.py.
#   docker exec symphony-master /opt/akida-client/run_client.sh --model kws_keyword_spotting_sparse --count 500
EGO_TOP=/opt/ibm/spectrumcomputing
SOAM_LIB="$EGO_TOP/soam/7.3.2/linux-x86_64/lib64"
source "$EGO_TOP/profile.platform" >/dev/null 2>&1
export PYTHONPATH="$SOAM_LIB/pythonapi_3.6.7:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$SOAM_LIB:$EGO_TOP/soam/7.3.2/linux-x86_64/lib:${LD_LIBRARY_PATH:-}"
# Cap the session to the fleet's chip count (AKIDA_NUM_NODES is set on the master
# by launch/up.sh) so one session uses one instance per chip; a user-supplied
# --max-services after this overrides it.
exec /usr/bin/python3.6 "$(dirname "$0")/soam_client.py" --max-services "${AKIDA_NUM_NODES:-0}" "$@"
