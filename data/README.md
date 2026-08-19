# Datasets (`data/`)

One folder per dataset, **exactly one `.npz` in each**, all committed through Git LFS. A clone
plus `git lfs pull` is everything the demos need: no downloads, no tfds, no `akida_models`
install, and no symlinks to set up.

| folder | `.npz` | feeds | samples | what it is |
|---|---|---|---|---|
| `speech_commands/` | `kws_keyword_spotting_sparse.npz` | `kws_keyword_spotting_sparse` | 4,890 | Google Speech Commands, as 49×10×1 MFCC frames |
| `coco2014_96/` | `vww_person_detect.npz` | `vww_person_detect` | 1,024 | Visual Wake Words (COCO2014 people), 96×96×3 |
| `voc2007/` | `voc2007_test_r448_100.npz` | `tiled_yolov2_voc` | 100 | VOC2007-test kit: frames, ground truth **and** reference detections |
| `surface_search_classifier/` | `random.npz` | `surface_search_classifier` | 256 | uniform noise; this model ships without real samples |

## Provenance and terms

Each set is derived from an upstream source that keeps its own terms: Google Speech
Commands (CC BY 4.0), Visual Wake Words from COCO 2014, and the PASCAL VOC 2007 test
split. `surface_search_classifier/` is seeded noise generated here, not real data. Full
attributions are in [NOTICE](../NOTICE); consult the upstream project before
redistributing derived data.

## How a dataset becomes a sample set

At launch, `src/common/prepare_samples.py` flattens every folder into a form the clients can
serve. The clients run under the master's Python 3.6 `soamapi` binding, which has no numpy, so
they cannot read `.npz` at all:

```
data/<dataset>/<one>.npz  ->  .cluster/shared/samples/<dataset>.bin           raw uint8, C-order
                              .cluster/shared/samples/<dataset>.samples.json  what it is
```

Two conventions carry all of that:

- **The folder names the set.** Not the file. Folder names are unique by construction, the
  output directory above is flat, and it is what makes "exactly one `.npz` per folder" a rule
  the code can enforce rather than a habit. A folder holding two `.npz` files is reported and
  skipped, because there is no honest way to name two sets after one folder.
- **The per-sample size names the model.** Nothing inside a file says which model it feeds;
  its sample shape does, matched against `models/<model>_meta.json` (448×448×3 → the shard
  model's `sample_input_shape`, 8×8×1 → `surface_search_classifier`, and so on). So a dataset
  can be called whatever describes it, and the wrong shape is caught rather than mis-served.

A dataset that cannot be prepared (unfetched, half-copied, wrongly shaped) is **reported and
skipped, never fatal**. One bad file must not take a cluster launch down with it.

## Git LFS

Every `.npz` here is an LFS object (`.gitattributes` tracks `*.npz`). A clone that skips the
pull gets ~130 bytes of pointer text in each file's place, which is a confusing thing to hit
deep inside a zip reader, so `up.sh`, `verify_reference.sh` and `prepare_samples.py` all
detect it and say the one thing worth saying:

```bash
git lfs install && git lfs pull
```

Everything under `data/` comes to about 87 MiB.

## `random.npz`: honest noise

`surface_search_classifier` maps `hw_only` on AKD1500, so it is a genuine on-chip
model-management demo, but no sample set ships with it. Rather than let each app improvise its
own random input, and disagree with the others about what is even runnable, the noise is
generated once, seeded, and committed like any other dataset.

The file declares itself synthetic with a `synthetic_random` marker **inside the `.npz`**, so
the flag travels with the bytes rather than with a filename that a copy or a `--name` could
rewrite. `prepare_samples.py` copies it into the sidecar as `"random": true`, and that is what
drives every "this is noise, not data" warning you see: the batch client's input line, the
workload dropdown label, and the banner above the results. Throughput, latency and the
per-chip split are all real; the predicted classes are meaningless and say so.

Drop real samples of the same shape into that folder and it stops being flagged.

## The VOC2007 kit

`voc2007_test_r448_100.npz` is a self-contained detection kit: 100 frames at 448×448×3, the
ground truth in **raw source pixels**, this model's own reference detections box for box, and
the full pipeline configuration (anchors, thresholds, tile geometry, merge parameters).
`scripts/verify_reference.py` cross-checks every one of those configuration fields against
`models/tiled_yolov2_voc_meta.json`, so a mismatched model/anchors pair is caught rather than
silently costing accuracy. The kit needs no companion file of any kind.

Carrying the reference detections is what makes the accuracy claim checkable rather than
merely stated: `scripts/eval_shard_map.py` prints what the fleet produced *and* the kit's own
stored detections scored by the same code. If the two rows agree, the pipeline is exact.

**Why 100 frames.** The full VOC2007 test split is 4,952 frames and 2.78 GiB, which is more than
a demo repo should ask of every clone. But an arbitrary 100 frames report an arbitrary mAP, so
these 100 were *chosen*: a uniform draw (seed 1234), rejected unless all 20 classes appear,
picked from 60,000 candidates as the one whose mAP50, mAP75 and mAP deviate least from the
full split's. It lands within 0.0016 on all three:

| | mAP50 | mAP75 | mAP |
|---|---|---|---|
| full split, 4,952 frames (published) | 0.4914 | 0.2112 | 0.2451 |
| this kit, 100 frames | 0.4909 | 0.2096 | 0.2450 |

The targets stored *in* the kit are its own, measured over its own frames by
`src/common/detection_map.py`, the same scorer that later checks a run against them. Read a
100-frame result as what it is, though: per-class AP over 100 images is noisy, and the number
to quote is still the published 49.14 on the whole split.

**Scoring the whole split.** The full kit stays outside the repo. Point the launcher at it:

```bash
./scripts/launch/up.sh image-shard-inference --nodes 6 --dataset ~/data/voc/VOCdevkit/voc2007_test_r448.npz
```

It is prepared as an extra set named after its own file, appears in the dashboard dropdown
alongside `voc2007`, and the client selects it with `--samples voc2007_test_r448`.

## Adding a dataset

Make a folder, put one `.npz` in it whose samples match a model's input shape, and relaunch.
Sample tensors are read from the largest `uint8` array with `ndim >= 2`, so a label vector
sitting alongside them is ignored rather than mistaken for data. That is all: `up.sh` prepares
it like the rest, and both dashboards offer it.
