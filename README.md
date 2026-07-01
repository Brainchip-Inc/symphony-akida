# symphony-akida

A turnkey **IBM Spectrum Symphony** (Community Edition 7.3.4) cluster in Docker
that runs **BrainChip Akida** models, with a runtime service that
loads / unloads / hot-swaps any `.fbz` model across the fleet, a management
console, and a laptop control dashboard + Python client.

This repo is BrainChip's fork of the partner-provided SymAkida demo, extended
to run inference on **real Akida silicon (AKD1000)** via device passthrough,
plus two extra demo models and their sample workloads.

> The full cluster runbook (stand up master + compute nodes, console access,
> HTTP API, teardown, known first-boot behavior) lives in
> **[BRAINCHIP-DEMO.md](BRAINCHIP-DEMO.md)** — start there.

---

## What this fork adds

- **On-chip execution.** The base image runs Akida on the software backend
  (`Model.forward()`). This fork runs each compute node against a physical
  AKD1000 by passing the device into the container (see below), and the
  dashboard now shows per-node/per-sample **ON-CHIP (AKD1000)** vs
  **SOFTWARE (CPU)** status.
- **Two extra models** in [`.models/`](.models/): a visual-wake-word person
  detector and a keyword-spotting model (quantized `iq8_wq4_aq4` `.h5`).
- **Sample workloads** in [`samples/`](samples/) for the dashboard's
  "Run across fleet" feature (person detect, keyword spotting, surface search).

---

## Repository layout

| Path | What |
|---|---|
| `BRAINCHIP-DEMO.md` | Full cluster runbook (authoritative) |
| `app.py`, `demo.py`, `akida_client.py` | Top-level dashboard + CLI + HTTP client |
| `web/` | Laptop control dashboard (Flask) — see [web/README.md](web/README.md) |
| `symakida-client/` | Standalone laptop client bundle — see [symakida-client/RUN.md](symakida-client/RUN.md) |
| `.models/` | Source Akida models (`.h5`) added by this fork |
| `samples/` | Bundled sample inputs (`.json`) for fleet workloads |

The 6.8 GB Docker image is **not** in this repo — pull it from the container
registry (next section).

---

## 1. Get the container image

The image is distributed via GitHub Container Registry (GHCR), not committed to
git. Replace `Brainchip-Inc` with the BrainChip GitHub org.

```bash
docker pull ghcr.io/Brainchip-Inc/symphony-akida:7.3.4
docker images ghcr.io/Brainchip-Inc/symphony-akida
# (optional) re-tag to the short name the runbook uses
docker tag ghcr.io/Brainchip-Inc/symphony-akida:7.3.4 symphonyce:7.3.4
```

If you're offline / have the tarball instead:

```bash
gunzip -c symphonyce-7.3.4-brainchip.tar.gz | docker load
```

---

## 2. Run the demo

Follow **[BRAINCHIP-DEMO.md §2](BRAINCHIP-DEMO.md)** to stand up 1 master +
3 compute nodes, then §3–§5 for console access, the HTTP API, and the laptop
dashboard.

### Running on real Akida hardware (this fork)

To execute on physical AKD1000s instead of the software backend, give each
compute container **one chip** via device passthrough — and drop `--privileged`
(passthrough does not need it). One chip per container:

```bash
# compute node N gets /dev/akidaN, presented in-container as /dev/akida0
docker run -d --network symcluster1 \
    --hostname symphony-compute-$n.local --name symphony-compute-$n \
    --network-alias symphony-compute-$n.local \
    -p 879$n:8790 \
    --device=/dev/akida$n:/dev/akida0 \
    -e HOST_ROLE=COMPUTE \
    -v /opt/symphony/shared:/shared \
    ghcr.io/Brainchip-Inc/symphony-akida:7.3.4
```

Verify a node is on-chip: the dashboard fleet status shows **ON-CHIP · AKD1000**
(green) for that node; `curl localhost:879N/health` reports the mapped device.

---

## 3. Models & samples

The `.fbz` models the service serves live in the containers' shared dir
(`/shared/models`, mounted from `/opt/symphony/shared`). To add a model from
this repo's `.h5` sources, convert it to a v1 `.fbz` with `cnn2snn` (inside the
image's akida env) and stage it into `/shared/models`; it then appears in
`GET /models` and the dashboard's model list. Sample inputs in `samples/` are
used by the dashboard's "Run across fleet" workload.

---

## Cloning this repo

Model (`*.h5`) and sample (`samples/*.json`) files are stored with **Git LFS**.
Install LFS before cloning or you'll get pointer stubs instead of the files:

```bash
sudo apt-get install -y git-lfs   # or: brew install git-lfs
git lfs install
git clone https://github.com/Brainchip-Inc/symphony-akida.git
```

If you already cloned without LFS: `git lfs pull`.
