"""Detection-merging worker (Python 3.12) for the shard pipeline's STITCH stage.

Runs as a subprocess of StitchServiceContainer and speaks the framed stdio protocol in
common/worker_io.py. Each request names one image whose six tiles have all been inferred; the
worker reads their per-tile detections off the shared pipeline bus, merges them into one
frame-level result and returns it.

The merge is the algorithm, not a tidy-up stage. Pooling the tiles and running a single global
per-class NMS -- the obvious thing, and what this app did before -- scores 22.70 mAP50 against
49.14 for the real merge: plain NMS cannot weld two halves of an object back together, and
cannot tell a duplicate from a fragment. Every parameter comes from the model's meta sidecar,
tuned on VOC trainval and never on a reporting split.

After the merge comes the one piece that is the app's own rather than the reference's: a score
gate on the *penalised* score. The per-tile confidence threshold runs before the merge, so it
cannot see the demotion the merge applies to fragments a seam still cuts off; without a gate
here a demoted half-object at 0.84 * 0.4 = 0.34 draws exactly like a complete detection at
0.87. Pass post_thresh 0 to reproduce the reference protocol for a mAP run.

Touches no Akida device, so this stage runs on the management host.
"""
import json
import os
import sys

import numpy as np

from tiled_shard import make_tile_layout, merge_tile_detections
from worker_io import log, serve

TAG = "shard-stitch-worker"
PIPE_DIR = os.environ.get("AKIDA_PIPELINE_DIR", "/shared/pipeline")
MODELS_DIR = os.environ.get("AKIDA_MODELS_DIR", "/shared/models")


class Stitcher:
    def __init__(self):
        self._cache = {}

    def config(self, model):
        if model not in self._cache:
            with open(os.path.join(MODELS_DIR, model + "_meta.json")) as handle:
                meta = json.load(handle)
            tiles = make_tile_layout(meta["tile_layout"], meta["frame_size"],
                                     meta["input_shape"][0])
            self._cache[model] = (meta, tiles)
            log(TAG, "%s: merging %d tiles into a %d frame, %s"
                % (model, len(tiles), meta["frame_size"], meta["merge"]))
        return self._cache[model]

    def __call__(self, request, _payload):
        model = request["model"]
        meta, tiles = self.config(model)
        num_classes = meta["num_classes"]
        max_boxes = request.get("max_boxes", meta["max_boxes"])
        post_thresh = request.get("post_thresh", meta["post_merge_thresh"])

        directory = os.path.join(PIPE_DIR, str(request["image_id"]))
        per_tile = []
        for index in range(len(tiles)):
            packed = np.load(os.path.join(directory, "det%d.npy" % index))
            per_tile.append((packed[:, :4], packed[:, 4], packed[:, 5].astype(np.int64),
                             packed[:, 6:6 + num_classes]))

        merged = merge_tile_detections(per_tile, tiles, meta["frame_size"],
                                       max_boxes=max_boxes, **meta["merge"])
        keep = merged.scores >= post_thresh
        boxes = merged.boxes[keep]
        scores = merged.scores[keep]
        labels = merged.labels[keep]
        truncated = merged.truncated[keep]

        names = meta["class_names"]
        histogram = {}
        for label in labels:
            name = names[label]
            histogram[name] = histogram.get(name, 0) + 1
        # Full float32 precision, not rounded. float32 -> float64 is exact and JSON round-trips
        # it exactly, whereas rounding to six decimals shifts coordinates by up to 5e-7. That is
        # invisible on screen but it is enough to flip a borderline IoU match, which shows up as
        # a few 1e-5 of mAP against the reference and makes an exact comparison impossible.
        return {"ok": True,
                "boxes": boxes.astype(float).tolist(),
                "scores": scores.astype(float).tolist(),
                "labels": labels.tolist(),
                "truncated": truncated.tolist(),
                "class_hist": histogram}


if __name__ == "__main__":
    sys.exit(serve(TAG, Stitcher()))
