#!/usr/bin/env bash
# Launch the Symphony + Akida cluster, sized to the host's Akida devices.
#
#   launch/up.sh            # one compute node per detected chip (capped at 7)
#
# One master + N compute (N = #chips, capped at the CE 64-core limit). Each
# compute owns one chip; the SOAM service preloads one on-chip instance per
# node. Everything is repo-local under .cluster/ (no /opt, no sudo).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-symphony-akida-demo:local}"
NETWORK="${NETWORK:-symcluster}"
SHARED="$HERE/.cluster/shared"
MODELS_SRC="$HERE/models"
MAX_NODES="${MAX_NODES:-7}"          # CE 64-core cap: master + 7 compute = 64 cores
CONSOLE_PORT="${CONSOLE_PORT:-8443}"

log(){ printf '\n[up] %s\n' "$*"; }
msh(){ docker exec symphony-master bash -lc "source /opt/ibm/spectrumcomputing/profile.platform >/dev/null 2>&1; egosh user logon -u Admin -x Admin >/dev/null 2>&1; $*"; }

# --- detect + health-probe chips -------------------------------------------
present=$(ls /dev/akida* 2>/dev/null | grep -Ec 'akida[0-9]+' || true)
[ "${present:-0}" -ge 1 ] || { echo "No /dev/akida* devices found; this demo needs Akida hardware." >&2; exit 1; }
log "found $present Akida chip node(s); probing health..."
mapfile -t HEALTHY < <(docker run --rm --privileged --entrypoint /opt/akida-service/probe_chips.sh "$IMAGE" 2>/dev/null)
total=${#HEALTHY[@]}
[ "$total" -ge 1 ] || { echo "No healthy Akida chips found. Try: sudo modprobe -r akida_pcie && sudo modprobe akida_pcie" >&2; exit 1; }
[ "$total" -lt "$present" ] && log "$((present-total)) chip(s) unhealthy -- skipping"
NODES=$total
if [ "$NODES" -gt "$MAX_NODES" ]; then
    log "capping at $MAX_NODES nodes (Symphony CE 64-core limit) -- $((total-MAX_NODES)) healthy chip(s) idle"
    NODES=$MAX_NODES
fi
CHIPS=("${HEALTHY[@]:0:$NODES}")
log "launching 1 master + $NODES compute node(s) on chips: ${CHIPS[*]}"

# --- clean slate ------------------------------------------------------------
for c in $(docker ps -aq --filter "name=symphony-master" --filter "name=symphony-compute-"); do docker rm -f "$c" >/dev/null; done
docker network rm "$NETWORK" >/dev/null 2>&1 || true
rm -rf "$HERE/.cluster"

# --- seed models ------------------------------------------------------------
mkdir -p "$SHARED/models"
ls "$MODELS_SRC"/*.fbz >/dev/null 2>&1 || { echo "No models in $MODELS_SRC (*.fbz). Run 'git lfs pull' first." >&2; exit 1; }
cp "$MODELS_SRC"/*.fbz "$MODELS_SRC"/*.json "$SHARED/models/" 2>/dev/null || true
log "seeded models: $(ls "$SHARED/models" | grep '\.fbz$' | tr '\n' ' ')"

# --- network + master -------------------------------------------------------
docker network create "$NETWORK" >/dev/null
log "starting master (console https://localhost:$CONSOLE_PORT/platform, Admin/Admin)"
docker run -d --privileged --name symphony-master \
    --network "$NETWORK" --hostname symphony-master.local --network-alias symphony-master.local \
    -e HOST_ROLE=MANAGEMENT -e AKIDA_NUM_NODES="$NODES" \
    -p "$CONSOLE_PORT:8443" \
    -v "$SHARED:/shared" \
    "$IMAGE" >/dev/null

log "waiting for the master's Session Director..."
for i in $(seq 1 60); do msh "soamview app >/dev/null 2>&1" && break; sleep 3; done

# --- compute nodes ----------------------------------------------------------
for j in $(seq 0 $((NODES-1))); do
    chip="${CHIPS[$j]}"
    docker run -d --privileged --name "symphony-compute-$j" \
        --network "$NETWORK" --hostname "symphony-compute-$j.local" --network-alias "symphony-compute-$j.local" \
        -e HOST_ROLE=COMPUTE -e AKIDA_CHIP="$chip" \
        --device="/dev/akida$chip" \
        -v "$SHARED:/shared" \
        "$IMAGE" >/dev/null
done

log "waiting for all $NODES compute node(s) to join..."
want=$((NODES+1))
for i in $(seq 1 60); do
    ok=$(msh "egosh resource list 2>/dev/null" | awk 'NR>1 && $2=="ok"' | wc -l) || true
    [ "${ok:-0}" -ge "$want" ] && { log "all $want hosts ok"; break; }
    sleep 3
done

# --- register + enable ------------------------------------------------------
log "registering + enabling the service (one instance per chip)"
docker exec symphony-master /opt/akida-service/register.sh "$NODES" 2>&1 | sed 's/^/    /'

log "waiting for $NODES instance(s) to map on-chip..."
for i in $(seq 1 40); do
    n=$(grep -l "worker READY" "$SHARED"/soam/akida-service/logs/si-*.log 2>/dev/null | wc -l) || true
    [ "${n:-0}" -ge "$NODES" ] && break
    sleep 3
done

log "service instance placement:"
msh "soamview service AkidaGenericService 2>&1" | sed 's/^/    /'
log "cluster up. Tear down with: launch/down.sh"
