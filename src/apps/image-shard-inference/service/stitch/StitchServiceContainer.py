"""SOAM ServiceContainer (Python 3.6) for the shard pipeline's STITCH stage.

Receives one image id whose six tiles have all been inferred and hands it to the python3.12
worker, which reads their per-tile detections off the shared pipeline bus
(/shared/pipeline/<image_id>/det{k}.npy), merges them into one frame-level result and returns
it. Boxes come back normalised to the frame, so the caller scales them however it needs:
by 448 to draw on the frame the fleet saw, by each image's raw shape to score against ground
truth.

No akida here, so this stage runs on the management host and never touches a chip.
"""
from __future__ import print_function
import json
import os

import soamapi

from shard_wire import PipeMessage, Py312Worker  # co-located in the deploy dir

DEFAULT_MODEL = os.environ.get("AKIDA_DEFAULT_MODEL", "tiled_yolov2_voc").strip()


class StitchServiceContainer(soamapi.ServiceContainer):
    def __init__(self):
        super(StitchServiceContainer, self).__init__()
        self._worker = Py312Worker("shard-stitch")

    def on_create_service(self, service_context):
        self._worker.start(want_shm=False)

    def on_session_enter(self, session_context):
        pass

    def on_invoke(self, task_context):
        in_msg = PipeMessage()
        task_context.populate_task_input(in_msg)
        image_id = str(in_msg.header.get("image_id", ""))
        request = {"image_id": image_id,
                   "model": in_msg.header.get("model") or DEFAULT_MODEL}
        for key in ("post_thresh", "max_boxes"):
            if in_msg.header.get(key) is not None:
                request[key] = in_msg.header[key]
        try:
            reply = self._worker.request(request)
            reply["image_id"] = image_id
            reply["n_boxes"] = len(reply.get("boxes", []))
        except Exception as exc:
            reply = {"image_id": image_id, "ok": False,
                     "error": "%s: %s" % (type(exc).__name__, exc)}
        out = soamapi.DefaultTextMessage()
        out.set_text(json.dumps(reply))
        task_context.set_task_output(out)

    def on_session_leave(self):
        pass

    def on_destroy_service(self):
        self._worker.stop()


if __name__ == "__main__":
    StitchServiceContainer().run()
