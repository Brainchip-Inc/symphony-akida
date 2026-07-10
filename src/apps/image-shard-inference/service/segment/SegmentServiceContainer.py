"""SOAM ServiceContainer (Python 3.6) for the shard pipeline's SEGMENT stage.

Receives a full 448x448x3 image as a binary task and splits it into five 224x224x3 segments
(four quadrants + overlapping center), writing each to the shared pipeline bus
(/shared/pipeline/<image_id>/seg{k}.bin) that the InferenceService reads. Only a tiny JSON
ack goes back over SOAM -- big tensors travel through /shared, not through the client.

Pure stdlib: sharding is byte-slicing (see shard_common.shard); no numpy/akida here, so this
stage runs on the management host and never touches an Akida chip.
"""
from __future__ import print_function
import array
import json
import os
import sys

import soamapi

from shard_common import shard, SEGMENTS  # co-located in the deploy dir

PIPE_DIR = os.environ.get("AKIDA_PIPELINE_DIR", "/shared/pipeline")


class PipeMessage(soamapi.Message):
    """Shard pipeline wire format (MUST match shard_client.py + the other containers):
       write_string(header_json); write_byte_array(array('B', payload), 0, len)."""

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


class SegmentServiceContainer(soamapi.ServiceContainer):
    def _log(self, msg):
        print("[shard-segment %d] %s" % (os.getpid(), msg), file=sys.stderr)
        sys.stderr.flush()

    def on_create_service(self, service_context):
        try:
            os.makedirs(PIPE_DIR, exist_ok=True)
        except Exception as e:
            self._log("could not create %s: %s" % (PIPE_DIR, e))
        self._log("ready; pipeline dir=%s" % PIPE_DIR)

    def on_session_enter(self, session_context):
        pass

    def on_invoke(self, task_context):
        in_msg = PipeMessage()
        task_context.populate_task_input(in_msg)
        image_id = str(in_msg.header.get("image_id", ""))
        img = in_msg.payload
        try:
            segs = shard(img)
            d = os.path.join(PIPE_DIR, image_id)
            os.makedirs(d, exist_ok=True)
            for k, seg in enumerate(segs):
                with open(os.path.join(d, "seg%d.bin" % k), "wb") as fh:
                    fh.write(seg)
            reply = {"image_id": image_id, "ok": True, "n_segments": len(segs),
                     "seg_names": [s[0] for s in SEGMENTS]}
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
    SegmentServiceContainer().run()
