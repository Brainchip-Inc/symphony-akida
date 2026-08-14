# Test kits (`data/voc/`)

Drop a VOC2007-test `.npz` kit in here and the launcher picks it up. Every kit it finds becomes a
selectable sample set in the `image-shard-inference` dashboard, alongside the committed random
fallback set. Zero, one or a dozen kits all work: with none, the demo runs on random frames.

The kits are gigabytes, so they are **never committed** (`data/voc/*.npz` is gitignored). Symlink
them in, which costs nothing and keeps one copy on disk:

```bash
ln -s ~/data/voc/VOCdevkit/voc2007_test_r448_first500.npz data/voc/
ln -s ~/data/voc/VOCdevkit/voc2007_test_r448.npz          data/voc/
./scripts/launch/up.sh image-shard-inference --nodes 6
```

Set up the same symlinks on another machine and that machine's launcher offers the same sets.

## What counts as a kit

Any `.npz` whose samples match a model's input shape, **under any filename**. The set is named
after the file, so `voc2007_test_r448.npz` becomes the set `voc2007_test_r448`, and a renamed copy
becomes a set under the new name. Nothing inside the file needs to say which model it is for: the
per-sample size does that (448×448×3 → `tiled_yolov2_voc`).

A file that does not fit is **reported and skipped**, not fatal: a stray or half-copied `.npz`
cannot take a cluster launch down with it.

Two tiers of kit are useful, and both are auto-detected:

| kit carries | how it is used |
|---|---|
| frames only | throughput, per-chip fan-out, drawn predictions |
| frames + ground truth + reference detections | all of the above, plus mAP and an exactness check |

A full kit is what makes the accuracy numbers possible without tfds, a VOC download or an
`akida_models` install: it holds the 448 frames, the ground truth in raw source pixels, the model
configuration, and this model's own reference detections box for box.

## No anchor file goes here

The model's five anchors live in `models/tiled_yolov2_voc_meta.json`, which is the single source of
configuration for the whole pipeline; `models/tiled_yolov2_voc_anchors.pkl` is committed next to it
purely as provenance for where they came from. `scripts/verify_reference.py` cross-checks every
field of that meta, anchors included, against the configuration stored inside each kit, so a
mismatched model/anchors pair is caught rather than silently costing accuracy.

So a kit needs no companion file of any kind. Just the `.npz`.
