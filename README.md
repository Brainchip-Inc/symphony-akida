# Symphony + Akida — on-chip fleet inference

Distribute AI inference across a fleet of **BrainChip Akida** AKD1000 devices on an **IBM
Spectrum Symphony** (Community Edition) cluster. One master + one compute node per chip
(capped at 7 — the CE 64-core limit); each node maps the model onto its chip
(`hw_only=True`) and runs inference **on-silicon**.

## Two demos, one image

Both apps build from the same image (`symphony-akida-demo:local`) and the same cluster.
The launcher activates exactly one at a time — they never run in parallel. Run them
back-to-back to show the contrast:

| App | Transport | Dispatch | Effect | Guide |
|---|---|---|---|---|
| **batch-inference** | Symphony SOAM | concurrent fan-out | every chip busy at once | [guide →](src/apps/batch-inference/README.md) |
| **serial-http-round-robin** | plain HTTP | round-robin, one at a time | ~one chip busy at a moment | [guide →](src/apps/serial-http-round-robin/README.md) |

Each app's README is the full clone → build → launch walkthrough.

<details>
<summary><b>Setup (once — shared by both apps)</b></summary>

Run on the host with the Akida cards (`/dev/akida*` + the `akida_pcie` driver) and Docker.

```bash
git clone <repo-url> symphony-akida && cd symphony-akida
git lfs install && git lfs pull                    # model .fbz + sample .npz
curl -LsSf https://astral.sh/uv/install.sh | sh    # host tooling for the dashboards
uv sync
docker build -f docker/Dockerfile -t symphony-akida-demo:local .   # bakes BOTH app backends
```

Then open the app you want to run and follow its README. To switch demos, tear down and
bring the other up:

```bash
./launch/down.sh && ./launch/up.sh <batch-inference|serial-http-round-robin>
```
</details>

<details>
<summary><b>Repository layout</b></summary>

```
docker/     image (FROM the Symphony+Akida base) + entrypoint; bakes both app backends
launch/     up.sh <app> / down.sh — size the cluster to detected chips
models/     on-chip .fbz models (Git LFS)
data/       sample .npz sets (Git LFS)
src/
  common/   shared code: akida_chip (on-chip core), models allowlist, sample prep
  apps/
    batch-inference/          SOAM service + client + dashboard (concurrent)
    serial-http-round-robin/  per-node HTTP server + client + dashboard (serial)
```
</details>

<details>
<summary><b>Design constraints</b></summary>

- **Community Edition ≤ 64 cores** → master + 7 compute; an 8th chip idles.
- **On-chip only** — a node with no mappable Akida device is not used for work.
- **Repo-local** — everything under `.cluster/` (bind-mounted to `/shared`); no `/opt`, no host `sudo`.
- **The image is defined by `docker/Dockerfile`, not shipped as a binary** — clone and
  `docker build`; the base layer pulls from `ghcr.io/brainchip-inc/symphony-akida`.
</details>

## Contributing

Commit convention and hook install: [CONTRIBUTING.md](CONTRIBUTING.md).
