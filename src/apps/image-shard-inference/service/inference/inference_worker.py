"""Akida on-chip inference worker (Python 3.12) for the shard pipeline's INFERENCE stage.

Runs as a subprocess of InferenceServiceContainer and speaks the framed stdio protocol in
common/worker_io.py. Each request names one tile of one image on the shared pipeline bus; the
worker reads that 224x224x3 tile, runs it on its chip, decodes the raw output into boxes, and
writes them back for the stitch stage.

    {"image_id": <str>, "tile": <int>, "model": <str|null>}\n

Decoding happens here, on the device side, rather than in the stitch stage: it belongs with
whatever produced the potentials, and it shrinks what crosses the bus from a 7x7x125 grid to
the handful of boxes that survived the confidence threshold. Merging belongs with whoever
collects all six tiles, which is the stitch stage.

Device selection, the hw_only map and predict all live in the shared akida_chip core
(/opt/akida-common), identical to the batch worker.
"""
import json
import os
import sys
import time

import numpy as np

from akida_chip import Chip, select_device, _stem  # shared on-chip core (/opt/akida-common)
from tiled_shard import decode_tile
from worker_io import log, serve

TAG = "shard-infer-worker"
PIPE_DIR = os.environ.get("AKIDA_PIPELINE_DIR", "/shared/pipeline")
MODELS_DIR = os.environ.get("AKIDA_MODELS_DIR", "/shared/models")


class TileDetector:
    """One chip, one mapped model, and the decode parameters that model was trained with."""

    def __init__(self, chip):
        self.chip = chip
        self._meta_cache = {}

    def meta(self, model):
        if model not in self._meta_cache:
            with open(os.path.join(MODELS_DIR, model + "_meta.json")) as handle:
                self._meta_cache[model] = json.load(handle)
        return self._meta_cache[model]

    def __call__(self, request, _payload):
        image_id = str(request["image_id"])
        tile = int(request["tile"])
        model = request.get("model")
        if model and _stem(model) != self.chip.stem:
            self.chip.load(model)
        meta = self.meta(self.chip.stem)

        directory = os.path.join(PIPE_DIR, image_id)
        with open(os.path.join(directory, "seg%d.bin" % tile), "rb") as handle:
            crop = np.frombuffer(handle.read(), dtype=np.uint8)

        output, inference_us, identity = self.chip.predict_tile(crop)
        started = time.perf_counter()
        # Anchors are fed unchanged to every tile, the downscaled whole-frame one included:
        # a YOLO head encodes size in units of input pixels, so decoding already yields the
        # correct tile-normalised size. Rescaling them per tile costs 24.7 mAP50.
        boxes, scores, labels, classes = decode_tile(output, meta["anchors"],
                                                     meta["num_classes"], meta["obj_thresh"],
                                                     meta["nms_thresh"])
        decode_us = int((time.perf_counter() - started) * 1e6)

        # One float32 (N, 4+1+1+num_classes) array per tile: xyxy, score, label, class scores.
        packed = np.column_stack([boxes, scores, labels.astype(np.float32), classes]) \
            if len(boxes) else np.zeros((0, 6 + meta["num_classes"]), dtype=np.float32)
        np.save(os.path.join(directory, "det%d.npy" % tile), packed.astype(np.float32))

        reply = {"ok": True, "tile": tile, "n_boxes": int(len(boxes)),
                 "inference_us": inference_us, "decode_us": decode_us}
        reply.update(identity)
        return reply


def main():
    # Retry device acquisition: on an SI restart the just-killed worker may still hold the
    # chip's driver lock for a moment. Retrying a few seconds lets it release before we exit,
    # turning a would-be restart cascade into a brief hiccup.
    device = None
    for attempt in range(12):
        try:
            device = select_device()
            break
        except Exception as exc:
            log(TAG, "device not ready (attempt %d/12): %s" % (attempt + 1, exc))
            time.sleep(2)
    if device is None:
        log(TAG, "FATAL: no Akida device after retries; node cannot run on-chip inference")
        return 1

    chip = Chip(device)
    default = os.environ.get("AKIDA_DEFAULT_MODEL", "").strip()
    if default:
        try:
            chip.load(default)  # strict hw_only: raises if the model will not map on-chip
            log(TAG, "mapped default %s hw_only" % default)
        except Exception as exc:
            log(TAG, "FATAL: default model %r failed to map hw_only: %s" % (default, exc))
            return 1
    serve(TAG, TileDetector(chip))
    return 0


if __name__ == "__main__":
    sys.exit(main())
