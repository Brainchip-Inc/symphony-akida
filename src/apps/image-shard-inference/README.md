# App: image-shard inference across the Akida fleet

Take one **448×448×3** frame, split it into **six 224×224×3 tiles**, run each through the **same
model on a separate Akida chip in parallel**, then merge the six detection sets back into one
result for the whole frame.

This is not a throughput trick. A 448 YOLOv2 **cannot map to AKD1500 at all** — `conv_0` accepts
at most 256 in its second dimension — so sharding into 224 tiles is the only route to 448-class
accuracy on this part. On the full VOC2007 test split it is worth **+8.6 mAP50 over the best
single-device option**:

| deployable on AKD1500 | mAP50 |
|---|---|
| **6 × 224 tiles, tile-trained model, full merge** | **49.14** |
| purpose-trained single-device 224 model | 41.51 |
| the same weights, one whole 224 frame | 40.58 |
| 6 × 224 tiles, pooled + plain per-class NMS | 22.70 |
| 448 whole frame (does not fit the hardware) | 54.69 |

The whole pipeline is **three Symphony SOAM services** — the client only sends a frame and reads
back merged detections:

```
client (master, thin)                        SOAM services (scheduled by Symphony)
  send 448 frame ──────────────▶ ShardSegmentService   (mgmt host, CPU)  → 6 × 224 tiles on /shared
  send 6 tile refs ────────────▶ ShardInferenceService (one per chip)    → predict + decode each
  send frame ref ──────────────▶ ShardStitchService    (mgmt host, CPU)  → merge + threshold
  ◀── merged detections
```

Big tensors travel over the shared dir (`/shared/pipeline/<image_id>/`); only tiny references,
the input frame and the final detections cross SOAM.

## The six tiles

Tile order is part of the contract, not a detail: fusion refuses to pair two fragments from the
same tile, so a permuted order silently disables it.

| index | position | name | x0 | y0 | region | fed to the model as |
|---|---|---|---|---|---|---|
| 0 | 1st | `top_left` | 0 | 0 | 224 | plain crop |
| 1 | 2nd | `top_right` | 224 | 0 | 224 | plain crop |
| 2 | 3rd | `bottom_left` | 0 | 224 | 224 | plain crop |
| 3 | 4th | `bottom_right` | 224 | 224 | 224 | plain crop |
| 4 | 5th | `center` | 112 | 112 | 224 | plain crop |
| 5 | **6th** | `global` | 0 | 0 | **448** | **the whole frame resized to 224** |

The sixth tile carries most of the result. Quadrants plus centre alone score 35.69 against 44.59
for all six on held-out data, which lands *below* a single-device whole-frame run; the four
quadrants on their own score 14.74. Adding tiles is **not monotonic** — extra tiles contribute
confident false positives, and only the full complement lets fusion and containment suppression
clean them up. There is deliberately no cheaper 5-tile mode.

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

# 4. launch the cluster on six chips — one per tile
./launch/up.sh image-shard-inference --nodes 6

# 5. open the dashboard, pick a sample set, Run
uv run python src/apps/image-shard-inference/dashboard/app.py
#   then browse http://localhost:5001
```

Over SSH? Forward the port: `ssh -L 5001:localhost:5001 <user>@<host>` and open
`http://localhost:5001` locally.
</details>

<details>
<summary><b>Returning users (everything already installed)</b></summary>

```bash
./launch/up.sh image-shard-inference --nodes 6                     # six chips, one per tile
./launch/up.sh image-shard-inference --nodes all                   # every healthy chip (CE caps at 7)
uv run python src/apps/image-shard-inference/dashboard/app.py      # http://localhost:5001

# ...or drive it straight from the CLI (runs the orchestrator client inside the master):
docker exec symphony-master /opt/akida-shard-client/run_client.sh --count 200

./launch/down.sh                                                   # tear down + wipe .cluster/
```

`--nodes` defaults to **6** for this app (one chip per tile) and is always capped at the
Symphony CE 64-core limit of master + 7 compute.
</details>

<details>
<summary><b>Measuring accuracy on VOC2007 test</b></summary>

The reporting split is **PASCAL VOC2007 test, 4,952 images** — the last VOC test set with public
ground truth. It ships as a self-contained `.npz` test kit holding the 448 frames, the ground
truth, the model configuration *and* the reference detections of this exact model, so nothing
else is needed: no tfds, no VOC download, no `akida_models`.

The kit lives outside the repo because the full split is 2.8 GiB. Point the launcher at it:

```bash
./launch/up.sh image-shard-inference --nodes 6 \
    --dataset ~/data/voc/VOCdevkit/voc2007_test_r448_first500.npz     # 500 frames, quick
./launch/up.sh image-shard-inference --nodes 6 \
    --dataset ~/data/voc/VOCdevkit/voc2007_test_r448.npz              # 4,952 frames, the published figure

# run it and dump the merged detections (--post-thresh 0 = the reference protocol)
docker exec symphony-master /opt/akida-shard-client/run_client.sh \
    --samples voc2007_test_r448 --count 4952 --ordered --post-thresh 0 --dump

uv run python scripts/eval_shard_map.py --per-class
```

`--ordered` matters: it makes frame *i* be sample *i*, which is what lets a dump be paired with
ground truth. The dashboard passes it automatically.

The scorer prints three rows. **fleet** is what the chips produced; **reference** is the kit's
own stored detections scored by the same code; **published** is the recorded target. Read the
fleet row against the reference row — both go through one scorer, so any drift in scoring moves
them together and cannot be mistaken for a pipeline regression. (The reference row itself sits
about 1e-4 above the published target; that offset is upstream of this repo, and
`scripts/eval_shard_map.py` documents what was ruled out.)

| metric | full-split target |
|---|---|
| mAP50 | 0.4914 |
| mAP75 | 0.2112 |
| mAP over IoU 0.50:0.95 | 0.2451 |

Only the full 4,952 frames reproduce those numbers; the 500-frame kit stores `-1.0` for them on
purpose.
</details>

<details>
<summary><b>Checking the pipeline is exact</b></summary>

Before any of the above, `scripts/verify_reference.sh` runs the ported pipeline on one chip and
compares **every merged box, score, label and truncated flag** against the kit's reference
detections. It fails on the first frame that disagrees and names the exact box, rather than
surfacing later as a vague point or two of mAP — and if it passes on every frame, the mAP is
identical to the published one by construction.

```bash
scripts/verify_reference.sh --frames 500                                   # quick kit
scripts/verify_reference.sh --npz ~/data/.../voc2007_test_r448.npz --frames all
```

It also cross-checks `models/tiled_yolov2_voc_meta.json` against the configuration stored inside
the kit — anchors, labels, tile geometry, thresholds and all nine merge parameters — which is the
cheapest guard against a mismatched model/anchors pair.

Run it with the cluster **down**: it takes a chip for itself, and driving the same chip from two
processes invites a DMA wedge.

Reading a bad mAP, if it ever comes to that: near 22–23 means the merge is not running at all;
35–36 means the `truncated` flag is being lost; 40–44 means one merge parameter is off; below 20
is upstream of the merge — channel layout, `predict` scaling, anchors, or RGB vs BGR.
</details>

<details>
<summary><b>Model</b></summary>

`tiled_yolov2_voc`: 20-class PASCAL VOC YOLOv2 / AkidaNet, input 224×224×3, output 7×7×125
(5 anchors × (4 box + 1 objectness + 20 classes)), 18 layers. Trained **on tiles**, which is
where 7.6 of the 8.6 mAP50 gain over a single device comes from. Maps to AKD1500 with
`hw_only=True` as a single hardware sequence.

Ships as `models/tiled_yolov2_voc.fbz` (Git LFS) + `models/tiled_yolov2_voc_anchors.pkl` +
`models/tiled_yolov2_voc_meta.json`, the last carrying the tile layout, thresholds and every
merge parameter so nothing is transcribed by hand.

**The model and its anchors must travel together.** A checkpoint trained with regenerated
anchors decodes to nonsense against any other anchor set. These anchors stand for object sizes
of 27×39, 53×94, 93×97, 140×174 and 192×196 pixels — all of which fit inside a 224 tile, which
is why they were regenerated; the 448 checkpoint's anchors include a 311×336 box that cannot.

Three things about running it that each cost real accuracy to get wrong:

- **`predict`, never `forward`.** `predict` rescales the integer potentials into the float range
  the decode expects. The rescale is **per output channel** — 125 distinct scales spanning 554 to
  28,029 — so no single global constant can stand in for it, and `predict` costs nothing extra.
- **Anchors are fed unchanged to all six tiles**, the downscaled whole-frame one included. A YOLO
  head encodes size in units of input pixels, so decoding already yields the correct
  tile-normalised size. Halving them for the sixth tile looks algebraically plausible and costs
  **−24.7 mAP50**.
- **The merge is the algorithm.** Pooling the tiles and running one global per-class NMS scores
  22.70 against 49.14. Plain NMS cannot weld two halves of an object together, and cannot tell a
  duplicate from a fragment.

Requires **akida 2.19.2** — the `.fbz` is serialized by it and 2.19.1 refuses to deserialize it,
so `docker/Dockerfile` upgrades the runtime and the image must be rebuilt after pulling this
change.
</details>

<details>
<summary><b>The merge, in order</b></summary>

`src/common/tiled_shard.py`, ported from `akida_models` `feature/tiled-yolov2-r448`:

1. **Map** every tile-local box into frame-normalised coordinates, recording which *interior
   seams* cut it off. A side is an interior seam only when it is not a true frame border, so a
   box touching the edge of the image is never treated as a fragment. The `global` tile has no
   interior seams at all.
2. **Fuse seam fragments.** Two detections fuse when they share a class, come from *different*
   tiles, and are complementarily truncated along one axis. The pair is replaced by its
   **enclosing box on both axes** — a seam cuts the object, not just its box, so the two extents
   across the split belong to two different pieces rather than being two estimates of one edge.
   Fusion is transitive, so an object split across three tiles collapses in one pass.
3. **Containment suppression** at IoS 0.7, using intersection over the *victim's* area: a
   fragment covering a third of an object has IoU 0.33 with it but IoS 1.0. Only truncated boxes
   may be removed — a small object genuinely inside a larger one of the same class is real, and
   suppressing on containment alone cost 8.9 recall points.
4. **Per-class NMS** at IoU 0.5, which removes the ordinary duplicates from overlapping tiles.
5. **Clip**, **demote** the fragments fusion could not complete (`score *= 0.4`), sort, keep
   `max_boxes`.

Then one thing the reference does not do, because it only matters on screen: a **score gate after
the merge**, on the penalised score. `obj_threshold` is applied per tile, before the merge, so
the demotion has nothing left to filter; without a gate here a demoted half-object at
0.84 × 0.4 = 0.34 draws exactly like a complete detection at 0.87. Default 0.5, exposed as
`--post-thresh`, and set to 0 for a mAP run because that is how the reference measures.

The per-detection `truncated` flag drives three of those mechanisms and is invisible in the box
coordinates, which makes it the thing a rewrite loses; between them they are worth about
5 mAP50 and 8.9 recall points. `_index()` carries it through every index, mask, sort and
concatenation for exactly that reason.
</details>

<details>
<summary><b>How it works</b></summary>

- `launch/up.sh image-shard-inference --nodes 6` launches one master + one compute container per
  chip (AKD1500 preferred, one chip pinned per container), seeds `models/`, prepares the sample
  sets, then registers the three SOAM services: `ShardInferenceService` (one instance per chip,
  `select(!mg)` + `EqualFreeSlot`) and `ShardSegmentService` / `ShardStitchService` (CPU, on the
  management host, `select(mg)`).
- The client (`shard_client.py`, in the master) streams frames through the three stages as a
  bounded pipeline — one SOAM session each, wired submit→fetch→submit, capped by an in-flight
  semaphore so `/shared` only holds a few frames at once. Per-frame correlation is by an
  `image_id` tag echoed in every reply. It reads frames from the sample `.bin` on demand rather
  than preloading, which the 2.8 GiB full split needs.
- soamapi is python3.6-only and numpy/akida are python3.12-only, so **all three** stages are a
  python3.6 `ServiceContainer` driving a python3.12 worker over framed stdio (`shard_wire.py`).
  Segment and stitch need it because the whole-frame downscale and the merge are numpy; the
  alternative, hand-porting the merge to python3.6 list math, is exactly the rewrite that loses
  the `truncated` flag.
- Decoding happens on the **device** side, not in the stitch stage: it belongs with whatever
  produced the potentials, and it shrinks what crosses the bus from a 7×7×125 grid per tile to
  the handful of boxes that survived the threshold.
- There is no service→service chaining (SOAM CE has none without heavy nested sessions) — the
  thin client sequences the stages while doing no image math itself.
- Symphony console: `https://localhost:8443/platform` (Admin/Admin).
</details>

<details>
<summary><b>Sample data</b></summary>

Two kinds, and the difference matters for what you can read off a run.

`data/samples/tiled_yolov2_voc.npz` (Git LFS) is **random 448 noise** — the fallback that always
travels with the repo, so a fresh clone can demo the fleet with no external data. It exercises
every stage and every throughput number, but it contains no objects, so **an empty result is the
correct one** and accuracy is not evaluated. The dashboard says so in a banner. Regenerate with
`uv run python scripts/make_shard_samples.py`.

The **VOC2007 test kit** `.npz` files are the real thing; see *Measuring accuracy* above. Pass
one to `./launch/up.sh --dataset <npz>` and it becomes a named sample set the client selects with
`--samples <name>`.

Either way `src/common/prepare_samples.py` flattens the frames into
`/shared/samples/<set>.bin` plus a sidecar, and the numpy-free client streams whole frames as
binary — the same path the vww/kws sets use.
</details>

<details>
<summary><b>Performance</b></summary>

The client reports **frames/sec**, **chips used**, per-chip **tile distribution**, average
on-chip latency per tile, and the fleet **speedup** vs one chip. A single chip runs a frame's six
tiles serially (≈ 6 × tile latency); the fleet runs them in parallel across six chips, so
full-frame throughput approaches a single chip's 224 throughput — higher effective resolution at
roughly the cost of one 224 inference.

`max_boxes` is 10, the VOC evaluation protocol rather than a deployment choice. It binds on only
12% of frames, but a crowded scene will hit it.

Worth knowing when choosing demo footage: tiling clearly **hurts** classes whose instances fill
the frame (`aeroplane`, `train`) and clearly **helps** medium or numerous ones (`tvmonitor`,
`dog`, `horse`).
</details>

<details>
<summary><b>Troubleshooting</b></summary>

- **An inference instance never maps on-chip:** check its log —
  `docker exec symphony-compute-0 bash -lc 'tail -n 40 /shared/soam/shard-inference/logs/si-*.log'`.
- **Segment or stitch failed to start:** they run a worker subprocess too —
  `tail -n 40 .cluster/shared/soam/shard-cpu/logs/*.log`.
- **Segment/stitch didn't come up on the management host:** if CE won't place workload instances
  on `mg`, edit `service/*/Shard{Segment,Stitch}Service.xml` to `resReq="select(!mg)"` +
  `resourceGroupName="ComputeHosts"` (they'll co-locate with inference; CPU contention is
  negligible since inference is on-chip), rebuild, relaunch.
- **Chip stuck (DMA timeout):** `sudo modprobe -r akida-pcie && sudo modprobe akida-pcie`,
  relaunch. Avoid driving a chip from two processes at once — don't run
  `scripts/verify_reference.sh` while the cluster is up.
- **`Cannot deserialize the model: created with Akida 2.19.2`:** the image predates the akida
  upgrade. Rebuild it: `docker build -f docker/Dockerfile -t symphony-akida-demo:local .`
- **Fewer nodes than chips:** CE caps the cluster at 64 cores → master + 7 compute; extra chips
  idle. `--nodes` can only go down from there.
</details>
