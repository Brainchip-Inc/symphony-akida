# App: image-shard inference across the Akida fleet

Take one large **448×448×3** image, split it into **five 224×224×3 segments** (four quadrants +
an overlapping center), run each segment through the **same** model on a **separate Akida chip in
parallel**, then stitch the five detection outputs back into one result for the full image. A
single chip can't take a 448 input cheaply, so sharding across chips buys the throughput of a
224 model while covering a higher-resolution image.

The whole pipeline is **three Symphony SOAM services** — the client only sends an image and reads
back the stitched result; the split, the on-chip inference, and the box stitching all happen in
the services, scheduled across the cluster by Symphony:

```
client (master, thin)                         SOAM services (scheduled by Symphony)
  send 448 image ───────────────▶ ShardSegmentService   (mgmt host, CPU)  → 5×224 segments on /shared
  send 5 segment refs ───────────▶ ShardInferenceService (one per chip)    → raw 7×7×35 grid each
  send image ref ────────────────▶ ShardStitchService    (mgmt host, CPU)  → decode + NMS-merge
  ◀── final detections
```

Big tensors travel over the shared dir (`/shared/pipeline/<image_id>/`); only tiny references +
the input image + final detections cross SOAM. The inference stage uses the same python3.6
container + python3.12 akida worker + `/dev/shm` hand-off as the batch-inference app, and imports
the shared `Chip` on-chip core (`hw_only`, map mode 2 = `HwPr`) — it just returns the raw
detector grid (`Chip.forward_raw`) instead of an argmax class.

> Inputs are random (`data/samples/yolo_akidanet_voc.npz`), so detections are meaningless — this
> app is about **throughput/latency**, not accuracy.

<details open>
<summary><b>First run (fresh clone)</b></summary>

Run on the host with the Akida cards (`/dev/akd1500_*` and/or `/dev/akida*` + the `akida-pcie`
driver) + Docker. The launcher prefers AKD1500 chips, falling back to AKD1000/NSoC_v2.

```bash
# 1. clone + fetch the LFS model/sample files
git clone <repo-url> symphony-akida && cd symphony-akida
git lfs install && git lfs pull

# 2. install uv (host tooling) + sync the dashboard env
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# 3. build the image (bakes all three app backends)
docker build -f docker/Dockerfile -t symphony-akida-demo:local .

# 4. launch the cluster in shard mode — auto-sizes to healthy chips, capped at 7 (CE limit)
./launch/up.sh image-shard-inference

# 5. open the dashboard, set an image count, Run
FLASK_PORT=5001 uv run python src/apps/image-shard-inference/dashboard/app.py
#   then browse http://localhost:5001
```

Over SSH? Forward the port: `ssh -L 5001:localhost:5001 <user>@<host>` and open `http://localhost:5001` locally.
</details>

<details>
<summary><b>Returning users (everything already installed)</b></summary>

```bash
./launch/up.sh image-shard-inference                              # bring the fleet up (3 services)
uv run python src/apps/image-shard-inference/dashboard/app.py     # http://localhost:5001

# ...or drive it straight from the CLI (runs the orchestrator client inside the master):
docker exec symphony-master /opt/akida-shard-client/run_client.sh --count 200

./launch/down.sh                                                  # tear down + wipe .cluster/
```
</details>

<details>
<summary><b>Model</b></summary>

The app uses `yolo_akidanet_voc` (VOC car/person YOLO akidanet): input 224×224×3, output 7×7×35
(5 anchors × (5 box + 2 classes)). It ships as a hardware-runnable Akida model —
`models/yolo_akidanet_voc.fbz` (Git LFS) + `models/yolo_akidanet_voc_meta.json` (input/sample
shapes, map mode `HwPr`, anchors, class names). Each inference instance maps it `hw_only`
(map mode 2 = `HwPr`) on its AKD1500 chip. The model is surfaced only by this app's dashboard via
`SHARD_MODELS` in `src/common/models.py` (kept out of the classifier apps' `SHOWN_MODELS`).
</details>

<details>
<summary><b>How it works</b></summary>

- `launch/up.sh image-shard-inference` launches one master + one compute container per **healthy**
  chip (AKD1500 preferred, one chip pinned per container), seeds `models/` and the sample `.bin`,
  then registers the three SOAM services:
  `ShardInferenceService` (one instance per chip, `select(!mg)` + `EqualFreeSlot`), and
  `ShardSegmentService` / `ShardStitchService` (CPU, on the management host, `select(mg)`).
- The client (`shard_client.py`, in the master) streams images through the three stages as a
  bounded pipeline — one SOAM session each, wired submit→fetch→submit, capped by an in-flight
  semaphore so `/shared` only holds a few images at once. Per-image correlation is by an
  `image_id` tag echoed in every reply.
- Symphony schedules each stage across the cluster; the 5 segments of each image fan out across
  the Akida chips. There is no service→service chaining (SOAM CE has none without heavy nested
  sessions) — the thin client sequences the stages while doing no image math itself.
- Symphony console: `https://localhost:8443/platform` (Admin/Admin).
</details>

<details>
<summary><b>Sample data</b></summary>

`data/samples/yolo_akidanet_voc.npz` holds random 448×448×3 uint8 images (regenerate with
`uv run python scripts/make_shard_samples.py`). At launch, `src/common/prepare_samples.py`
flattens it to `/shared/samples/yolo_akidanet_voc.bin` (using the meta's `sample_input_shape`),
and the numpy-free client streams whole images as binary — the same path the vww/kws sets use.
</details>

<details>
<summary><b>Performance</b></summary>

The client reports **images/sec**, **chips used**, per-chip **segment distribution**, avg on-chip
latency/segment, and the fleet **speedup** vs one chip. A single chip would run an image's five
segments serially (≈ 5 × segment latency); the fleet runs them in parallel across chips, so
full-image throughput approaches a single chip's 224 throughput × (chips / 5) — higher effective
resolution at roughly the cost of one 224 inference.
</details>

<details>
<summary><b>Troubleshooting</b></summary>

- **An inference instance never maps on-chip:** check its log —
  `docker exec symphony-compute-0 bash -lc 'tail -n 40 /shared/soam/shard-inference/logs/si-*.log'`.
- **Segment/stitch didn't come up on the management host:** if CE won't place workload instances
  on `mg`, edit `service/*/Shard{Segment,Stitch}Service.xml` to `resReq="select(!mg)"` +
  `resourceGroupName="ComputeHosts"` (they'll co-locate with inference; CPU contention is
  negligible since inference is on-chip), rebuild, relaunch.
- **Chip stuck (DMA timeout):** `sudo modprobe -r akida-pcie && sudo modprobe akida-pcie`, relaunch.
- **Fewer nodes than chips:** CE caps the cluster at 64 cores → master + 7 compute; extra chips idle.
</details>
