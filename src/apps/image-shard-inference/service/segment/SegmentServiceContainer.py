"""SOAM ServiceContainer (Python 3.6) for the shard pipeline's SEGMENT stage.

Receives one full 448x448x3 frame as a binary task and hands it to the python3.12 worker,
which cuts it into the six model-ready tiles on the shared pipeline bus
(/shared/pipeline/<image_id>/seg{k}.bin) for the InferenceService to read. Only a tiny JSON
ack goes back over SOAM -- big tensors travel through /shared, not through the client.

No akida here, so this stage runs on the management host and never touches a chip.
"""
from __future__ import print_function
import json

import soamapi

from shard_wire import PipeMessage, Py312Worker  # co-located in the deploy dir


class SegmentServiceContainer(soamapi.ServiceContainer):
    def __init__(self):
        super(SegmentServiceContainer, self).__init__()
        self._worker = Py312Worker("shard-segment")

    def on_create_service(self, service_context):
        self._worker.start()

    def on_session_enter(self, session_context):
        pass

    def on_invoke(self, task_context):
        in_msg = PipeMessage()
        task_context.populate_task_input(in_msg)
        image_id = str(in_msg.header.get("image_id", ""))
        try:
            reply = self._worker.request({"image_id": image_id,
                                          "model": in_msg.header.get("model")},
                                         in_msg.payload)
            reply["image_id"] = image_id
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
    SegmentServiceContainer().run()
