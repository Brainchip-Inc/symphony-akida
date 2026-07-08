# Symphony + Akida — on-chip fleet inference demo

Distribute AI inference across a fleet of **BrainChip Akida** devices using an **IBM
Spectrum Symphony** (Community Edition) cluster.

Each compute node owns one Akida chip. A SOAM **service** maps the model onto the
silicon (`hw_only=True`) and runs inference **on-chip**; a SOAM **client** fans a batch
of inputs across every chip in parallel. Throughput scales with the number of devices —
the opposite of one chip serving inputs serially.

> **Why this exists.** The original demo ran inference on the CPU (software backend) and
> dispatched work serially from a laptop, bypassing Symphony — so only one chip was ever
> busy. This rebuild runs genuinely on-chip and schedules through Symphony's SOAM session
> manager, so the fleet's advantage is real and measurable.

<details>
<summary><b>Repository layout</b></summary>

```
docker/          our image: FROM the Symphony+Akida base, our entrypoint + service
launch/          up.sh / down.sh — size the cluster to detected Akida devices
models/          on-chip .fbz models (Git LFS)
scripts/         commit-msg hook + installer
src/
  common/        shared host tooling (sample generator, helpers)
  apps/
    batch-inference/   app 1: batch inference across the fleet (service + client + dashboard)
```

</details>

<details>
<summary><b>Constraints baked into the design</b></summary>

- Symphony **Community Edition is capped at 64 cores** — the launcher sizes the cluster to
  the detected Akida devices and pins per-node `ncpus` to stay under the cap.
- Inference requires a real Akida device: a node with no mappable device is **not**
  available for work (strict on-chip rule).
- Everything is repo-local (no `/opt`, no host `sudo`).

</details>

<details>
<summary><b>Quick start</b></summary>

See the per-app guide: [`src/apps/batch-inference/README.md`](src/apps/batch-inference/README.md).
In short — install `uv`, `git lfs pull`, build the image, run `launch/up.sh`, open the
dashboard.

</details>

## Contributing

Commit convention and hook install: [CONTRIBUTING.md](CONTRIBUTING.md).
