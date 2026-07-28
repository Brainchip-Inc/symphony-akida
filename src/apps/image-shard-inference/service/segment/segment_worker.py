"""Tile-splitting worker (Python 3.12) for the shard pipeline's SEGMENT stage.

Runs as a subprocess of SegmentServiceContainer (soamapi is python3.6-only, numpy is not) and
speaks the framed stdio protocol in common/worker_io.py. Each request carries one 448x448x3
uint8 frame in the shared buffer; the worker cuts it into the six model-ready tiles and writes
them to the shared pipeline bus for the inference stage to pick up.

Five of the six tiles are plain crops. The sixth is the whole frame downscaled to the model
input, and it is not optional: it carries most of the accuracy, and quadrants-plus-centre
alone scores below a single-device whole-frame run.

Touches no Akida device, so this stage runs on the management host.
"""
import json
import os
import sys

import numpy as np

from tiled_shard import make_tile_layout, split_frame
from worker_io import log, serve

TAG = "shard-segment-worker"
PIPE_DIR = os.environ.get("AKIDA_PIPELINE_DIR", "/shared/pipeline")
MODELS_DIR = os.environ.get("AKIDA_MODELS_DIR", "/shared/models")


class Segmenter:
    def __init__(self):
        self._cache = {}

    def layout(self, model):
        """(tiles, frame_size, input_size) for a model, from its meta sidecar."""
        if model not in self._cache:
            with open(os.path.join(MODELS_DIR, model + "_meta.json")) as handle:
                meta = json.load(handle)
            frame_size = meta["frame_size"]
            input_size = meta["input_shape"][0]
            tiles = make_tile_layout(meta["tile_layout"], frame_size, input_size)
            self._cache[model] = (tiles, frame_size, input_size)
            log(TAG, "%s: %d tiles %s, frame %d -> input %d"
                % (model, len(tiles), [t.name for t in tiles], frame_size, input_size))
        return self._cache[model]

    def __call__(self, request, payload):
        image_id = str(request["image_id"])
        tiles, frame_size, input_size = self.layout(request["model"])
        frame = np.frombuffer(payload, dtype=np.uint8).reshape(frame_size, frame_size, 3)
        crops = split_frame(frame, tiles, input_size)

        directory = os.path.join(PIPE_DIR, image_id)
        os.makedirs(directory, exist_ok=True)
        for index, crop in enumerate(crops):
            with open(os.path.join(directory, "seg%d.bin" % index), "wb") as handle:
                handle.write(np.ascontiguousarray(crop).tobytes())
        return {"ok": True, "n_tiles": len(tiles), "tile_names": [t.name for t in tiles]}


if __name__ == "__main__":
    sys.exit(serve(TAG, Segmenter()))
