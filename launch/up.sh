#!/usr/bin/env bash
# Launch the Symphony + Akida cluster, sized to the host's Akida devices.
#
#   launch/up.sh [APP]      # APP = batch-inference (default) | serial-http-round-robin
#
# One master + N compute (N = #chips, capped at the CE 64-core limit). Each compute
# owns one chip. Both apps share this cluster and the image; the launcher activates
# exactly ONE backend per run (so only one process ever owns each chip -- and the two
# demos never run in parallel):
#   batch-inference          -> register the SOAM service (one on-chip instance per node);
#                               the dashboard triggers a concurrent SOAM fan-out.
#   serial-http-round-robin  -> each compute publishes host port 8790+j and runs a plain
#                               HTTP inference server; the dashboard round-robins /infer.
# Everything is repo-local under .cluster/ (no /opt, no sudo).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP="${1:-batch-inference}"
case "$APP" in
    batch-inference|serial-http-round-robin|image-shard-inference) ;;
    *) echo "unknown app: '$APP' (use: batch-inference | serial-http-round-robin | image-shard-inference)" >&2; exit 1;;
esac
IMAGE="${IMAGE:-symphony-akida-demo:local}"
NETWORK="${NETWORK:-symcluster}"
SHARED="$HERE/.cluster/shared"
MODELS_SRC="$HERE/models"
MAX_NODES="${MAX_NODES:-7}"          # CE 64-core cap: master + 7 compute = 64 cores
CONSOLE_PORT="${CONSOLE_PORT:-8443}"
PORT_BASE="${AKIDA_PORT_BASE:-8790}"      # serial-http: compute-j -> host port PORT_BASE+j
SHM_BYTES="${AKIDA_SHM_BYTES:-8388608}"   # per-node shared input buffer (8 MiB); raise for bigger models
SHM_SIZE="${AKIDA_SHM_SIZE:-128m}"        # docker /dev/shm size (must be >= SHM_BYTES)

log(){ printf '\n[up] %s\n' "$*"; }
msh(){ docker exec symphony-master bash -lc "source /opt/ibm/spectrumcomputing/profile.platform >/dev/null 2>&1; egosh user logon -u Admin -x Admin >/dev/null 2>&1; $*"; }

# --- detect + health-probe chips -------------------------------------------
# Two families of chip nodes: AKD1500 (/dev/akd1500_<N>, preferred) and AKD1000/NSoC_v2
# (/dev/akida<N>). probe_chips.sh returns healthy /dev node names, AKD1500 first.
present=$(ls -d /dev/akd1500_* /dev/akida[0-9]* 2>/dev/null | grep -Ec 'akd1500_[0-9]+|akida[0-9]+' || true)
[ "${present:-0}" -ge 1 ] || { echo "No Akida devices (/dev/akd1500_* or /dev/akida*) found; this demo needs Akida hardware." >&2; exit 1; }
log "found $present Akida chip node(s); probing health (AKD1500 preferred)..."
mapfile -t HEALTHY < <(docker run --rm --privileged --entrypoint /opt/akida-service/probe_chips.sh "$IMAGE" 2>/dev/null)
total=${#HEALTHY[@]}
[ "$total" -ge 1 ] || { echo "No healthy Akida chips found. Try: sudo modprobe -r akida-pcie && sudo modprobe akida-pcie" >&2; exit 1; }
[ "$total" -lt "$present" ] && log "$((present-total)) chip(s) unhealthy -- skipping"
NODES=$total
if [ "$NODES" -gt "$MAX_NODES" ]; then
    log "capping at $MAX_NODES nodes (Symphony CE 64-core limit) -- $((total-MAX_NODES)) healthy chip(s) idle"
    NODES=$MAX_NODES
fi
CHIPS=("${HEALTHY[@]:0:$NODES}")   # /dev node names, e.g. akd1500_0 akd1500_1 ...
log "app=$APP: launching 1 master + $NODES compute node(s) on chips: ${CHIPS[*]}"

# --- clean slate ------------------------------------------------------------
for c in $(docker ps -aq --filter "name=symphony-master" --filter "name=symphony-compute-"); do docker rm -f "$c" >/dev/null; done
docker network rm "$NETWORK" >/dev/null 2>&1 || true
rm -rf "$HERE/.cluster"

# --- seed models ------------------------------------------------------------
mkdir -p "$SHARED/models"
mkdir -p "$SHARED/pipeline" && chmod 777 "$SHARED/pipeline"  # image-shard-inference: per-image segment/grid
                                # bus (/shared/pipeline); world-writable so the container's egoadmin (uid
                                # 1000) can create per-image dirs regardless of the host user launching up.sh
ls "$MODELS_SRC"/*.fbz >/dev/null 2>&1 || { echo "No models in $MODELS_SRC (*.fbz). Run 'git lfs pull' first." >&2; exit 1; }
cp "$MODELS_SRC"/*.fbz "$MODELS_SRC"/*.json "$SHARED/models/" 2>/dev/null || true
log "seeded models: $(ls "$SHARED/models" | grep '\.fbz$' | tr '\n' ' ')"

# --- seed real samples (npz -> /shared/samples: <model>.bin + sidecar) ------
# Numpy-free client reads these; a model with no .npz falls back to random input.
if ls "$HERE/data/samples"/*.npz >/dev/null 2>&1; then
    if command -v uv >/dev/null 2>&1; then
        if ( cd "$HERE" && uv run python src/common/prepare_samples.py --out "$SHARED/samples" ) 2>&1 | sed 's/^/    /'; then
            log "seeded samples: $(ls "$SHARED/samples" 2>/dev/null | grep '\.bin$' | tr '\n' ' ')"
        else
            log "WARN: sample prep failed; client will use random inputs"
        fi
    else
        log "WARN: uv not found; skipping sample prep (client will use random inputs)"
    fi
fi

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
    node="${CHIPS[$j]}"
    extra=()
    # serial-http-round-robin: publish this node's HTTP server (container :8790) on
    # host port PORT_BASE+j and have the entrypoint start it (START_HTTP=1).
    if [ "$APP" = "serial-http-round-robin" ]; then
        extra+=( -p "$((PORT_BASE+j)):8790" -e START_HTTP=1 -e HTTP_PORT=8790 )
    fi
    docker run -d --privileged --name "symphony-compute-$j" \
        --network "$NETWORK" --hostname "symphony-compute-$j.local" --network-alias "symphony-compute-$j.local" \
        -e HOST_ROLE=COMPUTE -e AKIDA_CHIP_NODE="$node" -e AKIDA_SHM_BYTES="$SHM_BYTES" \
        --shm-size="$SHM_SIZE" \
        --device="/dev/$node" \
        -v "$SHARED:/shared" \
        ${extra[@]+"${extra[@]}"} \
        "$IMAGE" >/dev/null
done

log "waiting for all $NODES compute node(s) to join..."
want=$((NODES+1))
for i in $(seq 1 60); do
    ok=$(msh "egosh resource list 2>/dev/null" | awk 'NR>1 && $2=="ok"' | wc -l) || true
    [ "${ok:-0}" -ge "$want" ] && { log "all $want hosts ok"; break; }
    sleep 3
done

# --- app-specific backend ---------------------------------------------------
if [ "$APP" = "batch-inference" ]; then
    log "registering + enabling the SOAM service (one instance per chip)"
    docker exec symphony-master /opt/akida-service/register.sh "$NODES" 2>&1 | sed 's/^/    /'

    log "waiting for $NODES instance(s) to map on-chip..."
    for i in $(seq 1 40); do
        n=$(grep -l "worker READY" "$SHARED"/soam/akida-service/logs/si-*.log 2>/dev/null | wc -l) || true
        [ "${n:-0}" -ge "$NODES" ] && break
        sleep 3
    done

    log "service instance placement:"
    msh "soamview service AkidaGenericService 2>&1" | sed 's/^/    /'
    log "cluster up (batch-inference). Run the dashboard:"
    log "    uv run python src/apps/batch-inference/dashboard/app.py   # http://localhost:5001"
elif [ "$APP" = "image-shard-inference" ]; then
    log "registering + enabling the 3 shard services (segment=mgmt, inference=1/chip, stitch=mgmt)"
    docker exec symphony-master /opt/akida-shard-service/register.sh "$NODES" 2>&1 | sed 's/^/    /'

    log "waiting for $NODES inference instance(s) to map on-chip..."
    for i in $(seq 1 40); do
        n=$(grep -l "worker READY" "$SHARED"/soam/shard-inference/logs/si-*.log 2>/dev/null | wc -l) || true
        [ "${n:-0}" -ge "$NODES" ] && break
        sleep 3
    done
    log "waiting for the segment + stitch instance(s)..."
    for i in $(seq 1 20); do
        n=$(grep -l "ready;" "$SHARED"/soam/shard-cpu/logs/*.log 2>/dev/null | wc -l) || true
        [ "${n:-0}" -ge 2 ] && break
        sleep 3
    done

    log "service instance placement:"
    for app in ShardSegmentService ShardInferenceService ShardStitchService; do
        msh "soamview service $app 2>&1" | sed 's/^/    /'
    done
    log "cluster up (image-shard-inference). Run the dashboard:"
    log "    uv run python src/apps/image-shard-inference/dashboard/app.py   # http://localhost:5001"
    log "or drive it from the CLI:"
    log "    docker exec symphony-master /opt/akida-shard-client/run_client.sh --count 200"
else
    # serial-http-round-robin: each compute already started its HTTP server (START_HTTP=1).
    log "waiting for $NODES per-node HTTP server(s) to map on-chip..."
    for i in $(seq 1 40); do
        n=$(grep -l "listening on" "$SHARED"/soam/http-service/logs/http-*.log 2>/dev/null | wc -l) || true
        [ "${n:-0}" -ge "$NODES" ] && break
        sleep 3
    done
    log "per-node HTTP servers:"
    for j in $(seq 0 $((NODES-1))); do
        printf '    compute-%d -> http://localhost:%d\n' "$j" "$((PORT_BASE+j))"
    done
    log "cluster up (serial-http-round-robin). Run the dashboard:"
    log "    AKIDA_NODE_COUNT=$NODES uv run python src/apps/serial-http-round-robin/dashboard/app.py   # http://localhost:5001"
fi
log "Tear down with: launch/down.sh"
