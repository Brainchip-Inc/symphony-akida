# SymAkida — BrainChip demo runbook

A turnkey IBM Spectrum Symphony (Community Edition) cluster in Docker that
runs **BrainChip Akida** models, with a service that **loads / unloads /
hot-swaps any `.fbz` at runtime** across the fleet, a management console,
SSH, and a laptop dashboard.

Everything ships in one image: `symphonyce:7.3.4`.

---

## 0. What you get

| Piece | Where |
|---|---|
| Symphony CE 7.3.2 on Rocky 8 (glibc 2.28 so akida wheels load) | `symphonyce:7.3.4` image |
| BrainChip akida 2.19.1 SDK + Python 3.12 | baked in the image |
| Runtime model service (load/unload/hot-swap) — auto-registers on first boot | baked; serves HTTP per compute node |
| 3 demo `.fbz` models (+ class sidecars), seeded into the shared dir | baked |
| Management console (PMC) | `https://<host>:8443/platform` |
| SSH into a node | port 22 on the container (publish `-p 2222:22`) |
| Laptop control GUI + Python client | this repo: `web/`, `client/` |

> **Akida note:** these containers have no physical AKD1000, so inference
> runs on the akida **software backend** (`Model.forward()`); the service
> also maps each model to a virtual `AKD1500` as the "fits-on-silicon"
> proof. On real AKD1000/1500 hardware the same code runs on-chip.

---

## 1. Prerequisites

- Docker 20.10+ on Linux, ~8 GB RAM free.
- The image: either load the shipped tar **or** build from source.

```bash
# load the shipped image tar
gunzip -c symphonyce-7.3.4.tar.gz | docker load
docker images symphonyce:7.3.4
```

---

## 2. Stand up the cluster (1 master + 3 compute)

```bash
docker network create --driver bridge --subnet 172.30.0.0/24 symcluster1
sudo mkdir -p /opt/symphony/shared && sudo chown 1000:1000 /opt/symphony/shared

# master (publishes console 8443 and SSH 2222->22)
docker run -d --privileged --network symcluster1 \
    --hostname symphony-master.local --name symphony-master \
    --network-alias symphony-master.local \
    -p 8443:8443 -p 2222:22 \
    -e HOST_ROLE=MANAGEMENT \
    -e SSH_PUBLIC_KEY="$(cat ~/.ssh/id_rsa.pub)" \
    -v /opt/symphony/shared:/shared \
    symphonyce:7.3.4

# 3 compute nodes (publish each akida HTTP 8790 -> 879N)
for n in 1 2 3; do
  docker run -d --privileged --network symcluster1 \
    --hostname symphony-compute-$n.local --name symphony-compute-$n \
    --network-alias symphony-compute-$n.local \
    -p 879$n:8790 \
    -e HOST_ROLE=COMPUTE \
    -v /opt/symphony/shared:/shared \
    symphonyce:7.3.4
done
```

First boot does cluster init, seeds the demo models into `/shared/models`,
and auto-registers the akida service. The **console takes ~3–6 min** to
come up (Liberty is slow on the bundled 2021 JRE — this is expected).

> **Spreading across all chips.** The service auto-registers at master
> first-boot and comes up on the first compute node that joined; because
> it's a preStart service it does not rebalance as the other computes
> join. The image already bakes `ComputeHosts DistributeBy=EqualFreeSlot`,
> so once `egosh resource list` shows all compute nodes `ok`, one
> re-enable spreads it 1-per-host across the fleet:
>
> ```bash
> docker exec symphony-master bash -lc '
>   source /opt/ibm/spectrumcomputing/profile.platform
>   egosh user logon -u Admin -x Admin
>   echo Y | soamcontrol app disable AkidaGenericService; sleep 30
>   echo Y | soamcontrol app enable  AkidaGenericService'
> # then 8791, 8792, 8793 all serve (give the SIs ~1-2 min to spawn)
> ```
>
> Single-node (8791) works out of the box without this step.

---

## 3. Access

| What | How | Creds |
|---|---|---|
| Console (PMC) | `https://localhost:8443/platform` | `Admin` / `Admin` |
| SSH into master | `ssh egoadmin@localhost -p 2222` | the key you passed |
| Akida service | `http://localhost:8791` (compute-1; 8792/8793 for 2/3) | — |

---

## 4. Drive the akida service (HTTP)

```bash
curl localhost:8791/models                                   # list staged .fbz
curl -XPOST localhost:8791/load   -d '{"name":"voice_auth"}' # load on this node
curl localhost:8791/health                                   # current model
curl -XPOST localhost:8791/infer  -d '{"input":[7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7]}'
curl -XPOST localhost:8791/load   -d '{"name":"esm_classifier"}'  # hot-swap
curl -XPOST localhost:8791/unload
```

Endpoints: `GET /health`, `GET /models`, `POST /load|/reload|/unload|/infer`.
Drop more `.fbz` (and a `*_meta.json` / `*_params.json` class sidecar) into
`/opt/symphony/shared/models` and they appear in `/models`.

---

## 5. Laptop control GUI

Runs on the laptop, not the cluster — it calls the per-node HTTP endpoints.

```bash
git clone https://github.com/prop7/symakida && cd symakida
python3 -m venv web/.venv && ./web/.venv/bin/pip install flask
AKIDA_NODES="http://localhost:8791,http://localhost:8792,http://localhost:8793" \
FLASK_PORT=5001 ./web/.venv/bin/python web/app.py
# open http://localhost:5001
```

Dashboard: fleet status, load/unload/hot-swap, stage a local `.fbz`, and
**Run sample workload across the chips** (round-robins a bundled dataset
across the live nodes; shows per-sample node + latency + class histogram).
Python-only client: `client/akida_client.py`.

---

## 6. Teardown

```bash
for c in symphony-compute-3 symphony-compute-2 symphony-compute-1 symphony-master; do docker rm -f $c; done
docker network rm symcluster1
# fresh start: also wipe /opt/symphony/shared before re-launching the master
```

---

## Known issues & expected first-boot behavior

- **Akida SIs start on only one compute node.** On first boot the service
  auto-registers and enables at the master *before* the other compute
  nodes have finished joining, and because it's a **preStart** service
  (`taskLowWaterMark=0.0`, pinned, no reclaim) it does **not** rebalance
  as the rest of the fleet comes up. So immediately after first boot you
  will see `8791` serving but `8792`/`8793` down, with all SIs on
  `compute-1` (`soamview service AkidaGenericService` shows them stacked
  on one host). **This is expected, not a failure.** Single-node load /
  unload / infer works fine in this state. To spread 1-per-host, wait
  until `egosh resource list` shows every compute node `ok`, then run the
  one-shot disable/enable in §2 — `DistributeBy=EqualFreeSlot` is already
  baked, so the re-enable places one SI per chip. (Root cause history:
  the spread also requires the resource-plan `DistributeBy` to be
  `EqualFreeSlot`, which empty-by-default stacks greedily; that's baked
  into the image now.)
- **Console (8443) is slow to first paint** — ~3–6 min after first boot.
  Liberty installs its features for ~150s on the bundled 2021 JRE; the
  service heartbeat window is raised to 300s so the controller doesn't
  kill it mid-startup. A blank/refused 8443 in the first few minutes is
  normal.
- **Fresh `/shared` must be empty.** Re-running the master against a
  `/shared` that already has a cluster's `ego.conf` makes it take the
  recovery path with mismatched state. Wipe `/opt/symphony/shared` (or use
  a fresh docker volume) before re-launching from scratch.

## Build the image from source (optional)

Requires the IBM `.bin` installer + CE entitlement in
`khand-mesh/images/symphony/ctx/` (see `khand-mesh/docs/symphony-image-build.md`).

```bash
cd khand-mesh/images/symphony
DOCKER_BUILDKIT=1 docker build \
  --build-context python312=/opt/python3.12 \
  --build-context akida=/home/kjohnson/akida-venv \
  --build-context akida-service=/home/kjohnson/symakida/cluster-service \
  -t symphonyce:7.3.4 .
```
