#!/usr/bin/env bash
# Tear down the Symphony + Akida cluster and wipe repo-local state.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NETWORK="${NETWORK:-symcluster}"

removed=0
for c in $(docker ps -aq --filter "name=symphony-master" --filter "name=symphony-compute-"); do
    docker rm -f "$c" >/dev/null && removed=$((removed+1))
done
docker network rm "$NETWORK" >/dev/null 2>&1 || true
rm -rf "$HERE/.cluster"
echo "[down] removed $removed container(s), network '$NETWORK', and .cluster/"
