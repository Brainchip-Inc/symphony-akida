# symphony-akida

A turnkey **IBM Spectrum Symphony** (Community Edition 7.3.4) cluster in Docker
that runs **BrainChip Akida** models — with a runtime service that
loads / unloads / hot-swaps any `.fbz` across the fleet, a management console,
and a laptop control dashboard + Python client.

This is BrainChip's fork of the partner-provided SymAkida demo, extended to run
inference on **real Akida silicon (AKD1000)** via device passthrough, plus two
extra demo models and their sample workloads.

## ▶ Get started

**[QUICKSTART.md](QUICKSTART.md)** — install & run the demo in ~10 minutes
(with screenshots and copy-paste commands). Start here.

For the full reference — HTTP API, first-boot behavior, teardown, and
build-from-source — see **[BRAINCHIP-DEMO.md](BRAINCHIP-DEMO.md)**.

## What this fork adds

- **On-chip execution** — each compute node runs against a physical AKD1000 via
  device passthrough; the dashboard shows per-node **ON-CHIP (AKD1000)** vs
  **SOFTWARE (CPU)** status.
- **Two extra models** in [`.models/`](.models/) — a visual-wake-word person
  detector and a keyword-spotting model, shipped as ready-to-load `.fbz`
  (no conversion step needed; QUICKSTART stages them alongside the 3 models
  baked into the image).
- **Sample workloads** in [`samples/`](samples/) for the dashboard's
  "Run across fleet" feature.

## Repository layout

| Path | What |
|---|---|
| [QUICKSTART.md](QUICKSTART.md) | Install & run walkthrough — **start here** |
| [BRAINCHIP-DEMO.md](BRAINCHIP-DEMO.md) | Full cluster runbook (authoritative reference) |
| `app.py`, `demo.py`, `akida_client.py` | Top-level dashboard + CLI + HTTP client |
| [web/](web/) | Laptop control dashboard (Flask) — see [web/README.md](web/README.md) |
| [symakida-client/](symakida-client/) | Standalone laptop client bundle — see [symakida-client/RUN.md](symakida-client/RUN.md) |
| `.models/` | Two extra ready-to-load Akida models (`.fbz`) added by this fork |
| `samples/` | Bundled sample inputs (`.json`) for fleet workloads |

The 6.8 GB Docker image is **not** committed to git — it's distributed via
GitHub Container Registry (GHCR). QUICKSTART covers the pull.

## Cloning (Git LFS)

Model (`*.fbz`) and sample (`samples/*.json`) files are stored with **Git LFS** —
install it before cloning or you'll get pointer stubs:

```bash
sudo apt-get install -y git-lfs   # or: brew install git-lfs
git lfs install
git clone https://github.com/Brainchip-Inc/symphony-akida.git
```

Already cloned without LFS? Run `git lfs pull`.
