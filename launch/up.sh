#!/usr/bin/env bash
# Launch the Symphony + Akida cluster, sized to the host's Akida devices.
#
# One master + N compute, each owning one chip. All three apps share this cluster and the
# image; the launcher activates exactly ONE backend per run (so only one process ever owns
# each chip -- and the demos never run in parallel):
#   batch-inference          -> register the SOAM service (one on-chip instance per node);
#                               the dashboard triggers a concurrent SOAM fan-out.
#   serial-http-round-robin  -> each compute publishes host port 8790+j and runs a plain
#                               HTTP inference server; the dashboard round-robins /infer.
#   image-shard-inference    -> register the 3 shard services (segment/inference/stitch).
# Everything is repo-local under .cluster/ (no /opt, no sudo).
#
# Run `launch/up.sh --help` for the invocation, the flags and the environment overrides;
# usage() below is the single source for those, so they cannot drift from this comment.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
    cat <<'EOF'
Usage: launch/up.sh [APP] [--nodes N|all] [--dataset <npz>]
       launch/up.sh -h | --help

Launch the Symphony + Akida cluster: one master plus one compute node per Akida
chip, with exactly one app backend active. Needs Akida hardware on the host.

Apps
  batch-inference          Symphony SOAM, concurrent fan-out, every chip busy at
                           once. The default when APP is omitted.
  serial-http-round-robin  plain HTTP per node, round-robin, ~one chip at a time.
  image-shard-inference    Symphony SOAM in 3 stages: split one 448 frame into 6
                           tiles, infer in parallel, merge. Reports a real mAP.

Options
  --nodes N|all    how many chips to use. Defaults to 6 for image-shard-inference
                   (one per tile of a 448 frame) and to 'all' otherwise. Always
                   capped at the Community Edition 64-core limit of master + 7
                   compute, so 'all' on a bigger box leaves the extra chips idle.
  --dataset <npz>  image-shard-inference only: one extra test kit from outside
                   data/voc. Not needed for the usual case, since every kit in
                   data/voc is prepared automatically.
  -h, --help       show this and exit.

Sample sets
  The committed random set is always available, plus every .npz found in data/voc
  (symlink your test kits in there; see data/voc/README.md). All of them appear in
  the dashboard dropdown, and the client selects one with --samples <name>.

Environment overrides
  IMAGE            image to run                          (default symphony-akida)
  NETWORK          docker network name                   (default symcluster)
  NODES            fallback for --nodes                  (default per app, above)
  MAX_NODES        compute-node cap                      (default 7, the CE limit)
  CONSOLE_PORT     host port for the 8443 console        (default 8443)
  AKIDA_PORT_BASE  serial-http: compute-j -> this + j    (default 8790)
  AKIDA_SHM_BYTES  per-node shared input buffer, bytes   (default 8388608)
  AKIDA_SHM_SIZE   docker /dev/shm size, >= the above    (default 128m)
  AKIDA_KITS_DIR   where test kits are looked for        (default data/voc)

Examples
  launch/up.sh                                    batch-inference on every chip
  launch/up.sh image-shard-inference              6 chips, one per tile
  launch/up.sh image-shard-inference --nodes all  every chip, still 6 tiles
  launch/up.sh serial-http-round-robin --nodes 3  3 chips, ports 8790-8792

Tear the cluster down with launch/down.sh.
EOF
}

APP=""
WANT_NODES=""
DATASET=""
while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help) usage; exit 0;;
        --nodes)   WANT_NODES="${2:-}"; [ -n "$WANT_NODES" ] || { echo "--nodes needs a value (see --help)" >&2; exit 1; }; shift 2;;
        --dataset) DATASET="${2:-}";    [ -n "$DATASET" ]    || { echo "--dataset needs a value (see --help)" >&2; exit 1; }; shift 2;;
        -*) echo "unknown option: '$1' (see --help)" >&2; exit 1;;
        *) [ -z "$APP" ] || { echo "unexpected argument: '$1' (see --help)" >&2; exit 1; }; APP="$1"; shift;;
    esac
done
APP="${APP:-batch-inference}"
case "$APP" in
    batch-inference|serial-http-round-robin|image-shard-inference) ;;
    *) echo "unknown app: '$APP' (use: batch-inference | serial-http-round-robin | image-shard-inference; see --help)" >&2; exit 1;;
esac
# One chip per tile is the point of the shard demo, so it defaults to the tile count.
[ -n "$DATASET" ] && [ "$APP" != image-shard-inference ] && { echo "--dataset only applies to image-shard-inference (see --help)" >&2; exit 1; }
if [ -z "$WANT_NODES" ]; then
    [ "$APP" = image-shard-inference ] && WANT_NODES="${NODES:-6}" || WANT_NODES="${NODES:-all}"
fi
case "$WANT_NODES" in all|[1-9]|[1-9][0-9]) ;; *) echo "--nodes must be a positive number or 'all' (got '$WANT_NODES'; see --help)" >&2; exit 1;; esac
IMAGE="${IMAGE:-symphony-akida}"
NETWORK="${NETWORK:-symcluster}"
SHARED="$HERE/.cluster/shared"
MODELS_SRC="$HERE/models"
KITS_DIR="${AKIDA_KITS_DIR:-$HERE/data/voc}"   # test kits, symlinked in (data/voc/README.md)
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
if [ "$WANT_NODES" != all ] && [ "$WANT_NODES" -lt "$NODES" ]; then
    log "using $WANT_NODES of $total healthy chip(s) (--nodes) -- $((total-WANT_NODES)) idle"
    NODES=$WANT_NODES
elif [ "$WANT_NODES" != all ] && [ "$WANT_NODES" -gt "$total" ]; then
    log "WARN: --nodes $WANT_NODES requested but only $total healthy chip(s) found"
fi
if [ "$NODES" -gt "$MAX_NODES" ]; then
    log "capping at $MAX_NODES nodes (Symphony CE 64-core limit) -- $((total-MAX_NODES)) healthy chip(s) idle"
    NODES=$MAX_NODES
fi
CHIPS=("${HEALTHY[@]:0:$NODES}")   # /dev node names, e.g. akd1500_0 akd1500_1 ...
# probe_chips.sh returns AKD1500 first, so a mixed fleet only reaches AKD1000 once the
# AKD1500s run out. Worth saying out loud: the two families have different throughput, which
# makes a per-chip comparison misleading.
legacy=$(printf '%s\n' "${CHIPS[@]}" | grep -c '^akida[0-9]' || true)
[ "${legacy:-0}" -gt 0 ] && log "WARN: $legacy of $NODES chip(s) are AKD1000, not AKD1500"
log "app=$APP: launching 1 master + $NODES compute node(s) on chips: ${CHIPS[*]}"

# --- clean slate ------------------------------------------------------------
for c in $(docker ps -aq --filter "name=symphony-master" --filter "name=symphony-compute-"); do docker rm -f "$c" >/dev/null; done
docker network rm "$NETWORK" >/dev/null 2>&1 || true
"$HERE/launch/reclaim_shared.sh" "$HERE/.cluster/shared"
rm -rf "$HERE/.cluster"

# --- seed models ------------------------------------------------------------
mkdir -p "$SHARED/models"
# image-shard-inference: the per-frame tile/detection bus and the client's detection dumps.
# World-writable so the container's egoadmin (uid 1000) can write there regardless of which
# host user ran up.sh.
mkdir -p "$SHARED/pipeline" "$SHARED/results" && chmod 777 "$SHARED/pipeline" "$SHARED/results"
ls "$MODELS_SRC"/*.fbz >/dev/null 2>&1 || { echo "No models in $MODELS_SRC (*.fbz). Run 'git lfs pull' first." >&2; exit 1; }
cp "$MODELS_SRC"/*.fbz "$MODELS_SRC"/*.json "$SHARED/models/" 2>/dev/null || true
log "seeded models: $(ls "$SHARED/models" | grep '\.fbz$' | tr '\n' ' ')"

# --- seed sample sets (npz -> /shared/samples: <set>.bin + sidecar) ---------
# Numpy-free clients read these; a model with no set falls back to random input. Two sources:
# data/samples holds the small committed sets, so a fresh clone still demos; data/voc holds real
# test kits, which are gigabytes and therefore symlinked in rather than committed. Every kit
# found there becomes a sample set named after its own file, whatever that file is called.
prep() { ( cd "$HERE" && uv run python src/common/prepare_samples.py "$@" ) 2>&1 | sed 's/^/    /'; }
have_npz() { ls "$1"/*.npz >/dev/null 2>&1; }
if have_npz "$HERE/data/samples" || have_npz "$KITS_DIR" || [ -n "$DATASET" ]; then
    if ! command -v uv >/dev/null 2>&1; then
        log "WARN: uv not found; skipping sample prep (clients will use random inputs)"
    else
        prep --out "$SHARED/samples" || log "WARN: sample prep failed; clients will use random inputs"
        if [ "$APP" = image-shard-inference ] && have_npz "$KITS_DIR"; then
            log "preparing test kits from $(basename "$KITS_DIR")/ (a minute for the full split)"
            prep --kits "$KITS_DIR" --out "$SHARED/samples" \
                || log "WARN: test kit prep failed; the demo will run on random input"
        fi
        if [ -n "$DATASET" ]; then
            [ -f "$DATASET" ] || { echo "no such dataset: $DATASET" >&2; exit 1; }
            log "preparing test kit $(basename "$DATASET")"
            prep --npz "$DATASET" --out "$SHARED/samples" \
                || { echo "dataset prep failed" >&2; exit 1; }
        fi
        log "seeded sample sets: $(ls "$SHARED/samples" 2>/dev/null | grep '\.bin$' | sed 's/\.bin$//' | tr '\n' ' ')"
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

    # Count distinct HOSTS with a ready worker, not log files. register.sh bounces the app to
    # re-place its instances under EqualFreeSlot, so the pre-bounce round leaves behind logs
    # from instances that already exited -- counting files would satisfy the wait while the
    # real instances are still coming up.
    ready_hosts() { grep -l "worker READY" "$1"/*.log 2>/dev/null \
        | sed -E 's|.*/[a-z]+-||; s|-[0-9]+\.log$||' | sort -u | wc -l; }
    log "waiting for $NODES inference instance(s) to map on-chip..."
    for i in $(seq 1 40); do
        [ "$(ready_hosts "$SHARED/soam/shard-inference/logs")" -ge "$NODES" ] && break
        sleep 3
    done
    log "waiting for the segment + stitch instance(s)..."
    for i in $(seq 1 20); do
        n=$(grep -l "worker READY" "$SHARED"/soam/shard-cpu/logs/*.log 2>/dev/null | wc -l) || true
        [ "${n:-0}" -ge 2 ] && break
        sleep 3
    done

    log "service instance placement:"
    for app in ShardSegmentService ShardInferenceService ShardStitchService; do
        msh "soamview service $app 2>&1" | sed 's/^/    /'
    done
    log "cluster up (image-shard-inference) on $NODES chip(s) for 6 tiles per frame. Run the dashboard:"
    log "    uv run python src/apps/image-shard-inference/dashboard/app.py   # http://localhost:5001"
    log "or drive it from the CLI:"
    log "    docker exec symphony-master /opt/akida-shard-client/run_client.sh --count 200"
    # The scoring hint names the biggest set that carries ground truth, however it got prepared.
    best_set=""; best_n=0
    for side in "$SHARED"/samples/*.samples.json; do
        [ -f "$side" ] || continue
        grep -q '"has_ground_truth": *true' "$side" || continue
        n=$(sed -n 's/.*"count": *\([0-9]*\).*/\1/p' "$side")
        [ "${n:-0}" -gt "$best_n" ] && { best_n="$n"; best_set="$(basename "$side" .samples.json)"; }
    done
    if [ -n "$best_set" ]; then
        # --ordered so frame i is sample i, and --post-thresh 0 because the published mAP is
        # measured with no post-merge gate.
        log "or score a real test kit end to end:"
        log "    docker exec symphony-master /opt/akida-shard-client/run_client.sh \\"
        log "        --samples $best_set --count $best_n --ordered --post-thresh 0 --dump"
        log "    uv run python scripts/eval_shard_map.py"
    else
        log "NOTE: no test kit found, so the only sample set is random input -- every stage and"
        log "      every throughput number is exercised, but there is nothing to detect and no"
        log "      mAP. Symlink a kit into data/voc/ and relaunch (see data/voc/README.md)."
    fi
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
