"""SOAM ServiceContainer (Python 3.6) for the shard pipeline's STITCH stage.

Reads the five per-segment raw output grids for an image from the shared pipeline bus
(/shared/pipeline/<image_id>/grid{k}.bin), decodes each YOLO grid into boxes, offsets them
from segment-local pixels into the full 448 frame, and NMS-merges across all five into one
detection set for the whole image (see shard_common.stitch). Returns the detections + class
histogram as JSON over SOAM.

Pure stdlib (no numpy/akida): small-list math, so this stage runs on the management host.
Anchors/grid/classes come from the model meta sidecar in the shared models dir.
"""
from __future__ import print_function
import array
import json
import os
import sys

import soamapi

from shard_common import stitch, SEGMENTS  # co-located in the deploy dir

PIPE_DIR = os.environ.get("AKIDA_PIPELINE_DIR", "/shared/pipeline")
MODELS_DIR = os.environ.get("AKIDA_MODELS_DIR", "/shared/models")
DEFAULT_MODEL = os.environ.get("AKIDA_DEFAULT_MODEL", "yolo_akidanet_voc").strip()


class PipeMessage(soamapi.Message):
    """Wire format MUST match shard_client.py: write_string(header_json); write_byte_array(payload)."""

    def __init__(self, header=None, payload=b""):
        super(PipeMessage, self).__init__()
        self.header = header or {}
        self.payload = payload

    def on_serialize(self, stream):
        stream.write_string(json.dumps(self.header))
        arr = array.array("B", self.payload)
        stream.write_byte_array(arr, 0, len(arr))

    def on_deserialize(self, stream):
        self.header = json.loads(stream.read_string() or "{}")
        self.payload = stream.read_byte_array("B").tobytes()


class StitchServiceContainer(soamapi.ServiceContainer):
    def __init__(self):
        super(StitchServiceContainer, self).__init__()
        self._meta_cache = {}

    def _log(self, msg):
        print("[shard-stitch %d] %s" % (os.getpid(), msg), file=sys.stderr)
        sys.stderr.flush()

    def _meta(self, model):
        if model not in self._meta_cache:
            path = os.path.join(MODELS_DIR, model + "_meta.json")
            self._meta_cache[model] = json.load(open(path))
        return self._meta_cache[model]

    def on_create_service(self, service_context):
        self._log("ready; models=%s pipeline=%s" % (MODELS_DIR, PIPE_DIR))

    def on_session_enter(self, session_context):
        pass

    def on_invoke(self, task_context):
        in_msg = PipeMessage()
        task_context.populate_task_input(in_msg)
        image_id = str(in_msg.header.get("image_id", ""))
        model = in_msg.header.get("model") or DEFAULT_MODEL
        d = os.path.join(PIPE_DIR, image_id)
        try:
            meta = self._meta(model)
            grids = []
            for k in range(len(SEGMENTS)):
                with open(os.path.join(d, "grid%d.bin" % k), "rb") as fh:
                    a = array.array("i")
                    a.frombytes(fh.read())
                    grids.append(a.tolist())
            dets, hist = stitch(grids, meta)
            reply = {"image_id": image_id, "ok": True, "n_boxes": len(dets),
                     "detections": dets, "class_hist": hist}
        except Exception as e:
            reply = {"image_id": image_id, "ok": False,
                     "error": "%s: %s" % (type(e).__name__, e)}
        out = soamapi.DefaultTextMessage()
        out.set_text(json.dumps(reply))
        task_context.set_task_output(out)

    def on_session_leave(self):
        pass

    def on_destroy_service(self):
        pass


if __name__ == "__main__":
    StitchServiceContainer().run()
