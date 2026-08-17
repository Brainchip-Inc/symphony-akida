# App: serial HTTP round-robin across the Akida fleet

The "before" demo, restored and fixed. A laptop dashboard talks to a **plain HTTP
inference server on each compute chip** and dispatches one `/infer` at a time,
**round-robin** across the nodes, so roughly one chip is busy at a moment. It is the
deliberate contrast to the `batch-inference` app, which submits a single Symphony SOAM
session that Symphony fans **concurrently** across every chip. Run them one after the
other to show the difference.

Three things that were wrong in the original are fixed here:
- **On-chip for real**: each node maps the model with `hw_only=True` on its own chip
  (via the shared `akida_chip` core), so the **ON-CHIP** badge is truthful, and the device
  beside it is the one that node actually holds (`AKD1500` or `AKD1000`, read from the
  chip's own `HwVersion`, never a fixed string).
- **Allowlisted models only**: the model list is `SHOWN_MODELS` (`src/common/models.py`), and
  the dataset picker offers only those of them with a prepared sample set, so it never lists a
  workload it cannot actually run.
- **Real `.npz` samples**: the workload is fed from the same `.npz`-derived samples the
  batch app uses (`prepare_samples.py` → `<dataset>.bin`), not the old fat JSON int-lists.

Both apps share one image (`symphony-akida`) and one cluster. `scripts/launch/up.sh`
activates only one backend per run and tears any previous cluster down first, so the two
demos never run in parallel.

<details open>
<summary><b>First run (fresh clone)</b></summary>

Run on the host with the Akida cards (needs `/dev/akd1500_*` and/or `/dev/akida*` and the `akida_pcie` driver) + Docker.

```bash
# 1. clone and fetch the LFS model + sample files
git clone <repo-url> symphony-akida && cd symphony-akida
git lfs install && git lfs pull

# 2. install uv (host tooling) + sync the dashboard env
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# 3. build the image (shared by both apps): public sources only; slow the first
#    time, cached after that. ACCEPT_IBM_LICENSE is required and has no default:
#    see the repo README's Licensing section.
docker build --build-arg ACCEPT_IBM_LICENSE=yes -f docker/Dockerfile -t symphony-akida .
docker run --rm --entrypoint /usr/local/bin/verify-image symphony-akida --full

# 4. launch the cluster in serial-http mode: auto-sizes to healthy chips, capped at 7
./scripts/launch/up.sh serial-http-round-robin

# 5. open the dashboard -- sizes the node list to the chips up.sh actually launched
./src/apps/serial-http-round-robin/dashboard/run_dashboard.sh
#   then browse http://localhost:5001
```

Over SSH? Forward the port: `ssh -L 5001:localhost:5001 <user>@<host>` and open `http://localhost:5001` locally.
</details>

<details>
<summary><b>Returning users (everything already installed)</b></summary>

```bash
./scripts/launch/up.sh serial-http-round-robin                          # bring the fleet up (HTTP servers per chip)
./src/apps/serial-http-round-robin/dashboard/run_dashboard.sh   # http://localhost:5001

./scripts/launch/down.sh                                                # tear down + wipe .cluster/
```

Switching demos just means relaunching with the other app name; `up.sh` removes the
running cluster first:

```bash
./scripts/launch/down.sh
./scripts/launch/up.sh batch-inference     # the concurrent-SOAM demo
```
</details>

<details>
<summary><b>Models</b></summary>

Only the two models both apps surface (in `models/`, Git LFS):

| Model | Input | Classes |
|---|---|---|
| `kws_keyword_spotting_sparse` | 49×10×1 | 12 keywords |
| `vww_person_detect` | 96×96×3 | person / background |

The allowlist lives in `src/common/models.py` (`SHOWN_MODELS`) and is shared with the
batch app; widen it to expose more models in both UIs at once.
</details>

<details>
<summary><b>How it works</b></summary>

- `scripts/launch/up.sh serial-http-round-robin` launches one compute container per **healthy**
  chip, pins it to that chip, publishes its HTTP server on host port `8790+j`, and sets
  `START_HTTP=1` so the entrypoint starts `run_http_server.sh` on the node.
- Each `http_server.py` (Python 3.12, akida venv) maps the default model `hw_only=True` on
  its chip using the shared `Chip` class, then serves `/health /models /load /reload
  /unload /infer`. A node with no device answers `/health` with `hardware_present:false`
  (shown as "no chip"); a model that will not map hw_only falls back to software and the
  node honestly shows **SOFTWARE**.
- The dashboard (`app.py`, host/uv) probes every node, then on **Run across fleet** reads
  `<model>.bin`, and round-robins one HTTP `/infer` per sample: `nodes[i % len(nodes)]`.
- The SOAM service is **not** registered in this mode, so nothing else contends for the
  chips. Symphony console still available at `https://localhost:8443/platform` (Admin/Admin).
</details>

<details>
<summary><b>Sample data</b></summary>

Same sample sets as the batch app: `data/<dataset>/<one>.npz` (git LFS, one folder per
dataset, see [`data/README.md`](../../../data/README.md)). `up.sh` converts them to
`<dataset>.bin` + `<dataset>.samples.json` under `.cluster/shared/samples` (via
`src/common/prepare_samples.py`); the dashboard slices per-sample bytes and sends them as
the HTTP `{"input":[…]}` array. A dataset that is synthetic noise says so in its own sidecar
and the dropdown labels it, so a noise run never reads as a real one. Because this is the
serial "before" path, the run is capped at `AKIDA_SAMPLE_LIMIT` samples (default 200) so it
completes promptly; raise it to run more.
</details>

<details>
<summary><b>Troubleshooting</b></summary>

- **A node shows down / no chip:** check its HTTP log: `docker exec symphony-compute-0
  bash -lc 'tail -n 40 /shared/soam/http-service/logs/http-*.log'`, or `curl
  http://localhost:8790/health`.
- **Fewer nodes than chips:** Community Edition caps the cluster at 64 cores → master + 7
  compute; extra chips idle (logged at launch).
- **Chip stuck (DMA timeout):** `sudo modprobe -r akida_pcie && sudo modprobe akida_pcie`, relaunch.
- **Dashboard shows all nodes down:** confirm you launched with `serial-http-round-robin` --
  the batch mode does not publish per-node ports, so there is nothing to talk to. The
  dashboard reads the node list and its ports off the running containers, so it cannot
  disagree with `up.sh` any more; if docker is not reachable from where you run it, set
  `AKIDA_NODES` (explicit URLs) or `AKIDA_NODE_COUNT` (ports 8790, 8791, ...) yourself.
</details>
