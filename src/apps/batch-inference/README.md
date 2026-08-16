# App: batch inference across the Akida fleet

Submit a batch of inputs as a **single Symphony SOAM session**. Symphony's session
manager fans the tasks across **every Akida chip in parallel**, each running the
model mapped on-silicon (`hw_only`). The dashboard shows the per-chip distribution
and throughput — the multi-Akida advantage over one chip serving inputs serially.

<details open>
<summary><b>First run (fresh clone)</b></summary>

Run on the host with the Akida cards (it needs `/dev/akd1500_*` and/or `/dev/akida*` and the `akida_pcie` driver) + Docker.

```bash
# 1. clone and fetch the LFS model files
git clone <repo-url> symphony-akida && cd symphony-akida
git lfs install && git lfs pull

# 2. install uv (host tooling) + sync the dashboard env
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# 3. build the image (Symphony CE + Akida runtime + our service & client) — public
#    sources only; slow the first time, cached after that. ACCEPT_IBM_LICENSE is
#    required and has no default: see the repo README's Licensing section.
docker build --build-arg ACCEPT_IBM_LICENSE=yes -f docker/Dockerfile -t symphony-akida .
docker run --rm --entrypoint /usr/local/bin/verify-image symphony-akida --full

# 4. launch the cluster — auto-sizes to healthy chips, capped at 7 (CE limit)
./scripts/launch/up.sh

# 5. open the dashboard, pick a model, set a sample count, Run
FLASK_PORT=5001 uv run python src/apps/batch-inference/dashboard/app.py
#   then browse http://localhost:5001
```

Over SSH? Forward the port: `ssh -L 5001:localhost:5001 <user>@<host>` and open `http://localhost:5001` locally.
</details>

<details>
<summary><b>Returning users (everything already installed)</b></summary>

```bash
./scripts/launch/up.sh                                            # bring the fleet up
uv run python src/apps/batch-inference/dashboard/app.py   # http://localhost:5001

# ...or drive it straight from the CLI (runs the SOAM client inside the master):
docker exec symphony-master /opt/akida-client/run_client.sh --model kws_keyword_spotting_sparse --count 5000

./scripts/launch/down.sh                                          # tear down + wipe .cluster/
```
</details>

<details>
<summary><b>Models</b></summary>

Models that map `hw_only` on the fleet's chips (in `models/`, Git LFS), surfaced through the
shared allowlist `src/common/models.py`:

| Model | Input | Classes | Sample set |
|---|---|---|---|
| `kws_keyword_spotting_sparse` | 49×10×1 | 12 keywords | ✅ real |
| `vww_person_detect` | 96×96×3 | person / background | ✅ real |
| `surface_search_classifier` | 8×8×1 | 7 classes | ⚠️ committed noise |

`surface_search_classifier` ships without real samples, so `data/surface_search_classifier/`
holds seeded uniform noise instead. It runs everywhere the other two do and measures throughput
honestly; its class histogram is meaningless, and every app says so — the dataset marks itself
synthetic and the flag rides through to the client's input line and the dashboard banner. It
still maps `hw_only`, so it is a real on-chip model for the load/hot-swap demo. Replace that
folder's `.npz` with real 8×8×1 samples and the warnings go away.

Add more by dropping a chip-mappable `.fbz` (+ a `<name>_meta.json` with `input_shape`/`class_names`)
into `models/` and adding its stem to `SHOWN_MODELS` in `src/common/models.py`.
</details>

<details>
<summary><b>How it works</b></summary>

- `scripts/launch/up.sh` probes each chip, launches one compute container per **healthy** chip
  (skips stuck ones), pins it to that chip, waits for all to join, then registers and
  enables the SOAM service one-instance-per-chip.
- Each service instance maps the model on its chip with `hw_only=True`. **A node with no
  mappable Akida device does not become available** — inference only ever runs on-chip.
- `soam_client.py` (in the master) opens one session and submits one task per sample;
  the session manager distributes them across every chip.
- Symphony console: `https://localhost:8443/platform` (Admin/Admin).
</details>

<details>
<summary><b>Sample data</b></summary>

Sample sets ship as `data/<dataset>/<one>.npz` (git LFS, one folder per dataset — see
[`data/README.md`](../../../data/README.md)). At launch, `up.sh` converts each to a raw
`<dataset>.bin` + `<dataset>.samples.json` sidecar under `/shared/samples` (via
`src/common/prepare_samples.py`), and the client streams those tensors as **binary** — no
JSON int arrays. The client finds its set through the sidecars, by model, so the set is free
to be named after the data rather than after the model. With no set at all it falls back to
random uint8 generated inline, so it still runs.

```bash
# regenerate the /shared bin sidecars by hand (up.sh does this automatically)
uv run python src/common/prepare_samples.py --out .cluster/shared/samples
```
</details>

<details>
<summary><b>Troubleshooting</b></summary>

- **`No healthy Akida chips` / a chip is stuck** (DMA timeout): reset the driver, then relaunch —
  `sudo modprobe -r akida_pcie && sudo modprobe akida_pcie && ./scripts/launch/up.sh`.
- **Fewer nodes than chips:** Community Edition caps the cluster at 64 cores → master + 7 compute.
  Extra chips are left idle (logged at launch).
- **Console slow to first paint** (~3–6 min after first boot): expected on the bundled JRE.
- **Inspect instances:** `docker exec symphony-master bash -lc 'source /opt/ibm/spectrumcomputing/profile.platform; egosh user logon -u Admin -x Admin; soamview service AkidaGenericService'`
</details>
