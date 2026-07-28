"""SOAM ServiceContainer (Python 3.6) for the shard pipeline's INFERENCE stage.

One pre-started instance per Akida chip (like the batch-inference service). Each task names
one tile of one image on the shared pipeline bus; the python3.12 worker reads that tile, runs
it on this node's chip and writes the decoded boxes back for the StitchService. Only a tiny
JSON ack (host, on-chip latency, box count) returns over SOAM.

The tile itself never crosses this process: the worker reads /shared directly, so there is no
payload to pass and no shared buffer to allocate.
"""
from __future__ import print_function
import json
import threading

import soamapi

from shard_wire import PipeMessage, Py312Worker  # co-located in the deploy dir


class InferenceServiceContainer(soamapi.ServiceContainer):
    def __init__(self):
        super(InferenceServiceContainer, self).__init__()
        self._worker = Py312Worker("shard-infer")
        self._lock = threading.Lock()

    def on_create_service(self, service_context):
        self._worker.start(want_shm=False)

    def on_session_enter(self, session_context):
        pass

    def on_invoke(self, task_context):
        in_msg = PipeMessage()
        task_context.populate_task_input(in_msg)
        image_id = str(in_msg.header.get("image_id", ""))
        tile = int(in_msg.header.get("tile", 0))
        try:
            with self._lock:   # one chip, one request at a time
                reply = self._worker.request({"image_id": image_id, "tile": tile,
                                              "model": in_msg.header.get("model")})
            reply["image_id"] = image_id
            reply.setdefault("tile", tile)
        except Exception as exc:
            reply = {"image_id": image_id, "tile": tile, "ok": False,
                     "error": "%s: %s" % (type(exc).__name__, exc)}
        out = soamapi.DefaultTextMessage()
        out.set_text(json.dumps(reply))
        task_context.set_task_output(out)

    def on_session_leave(self):
        pass

    def on_destroy_service(self):
        self._worker.stop()


if __name__ == "__main__":
    InferenceServiceContainer().run()
