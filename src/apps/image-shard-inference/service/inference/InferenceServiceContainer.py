"""SOAM ServiceContainer (Python 3.6) for the shard pipeline's INFERENCE stage.

One pre-started instance per Akida chip (like the batch-inference service). Each task names a
segment on the shared pipeline bus; the container reads that 224x224x3 segment from
/shared/pipeline/<image_id>/seg{k}.bin, hands it to the python3.12 akida worker via a /dev/shm
buffer (soamapi is 3.6-only, akida is 3.12-only), and writes the raw output grid back to
/shared/pipeline/<image_id>/grid{k}.bin for the StitchService. Only a tiny JSON ack (host +
on-chip latency) returns over SOAM.

The shm hand-off + worker lifecycle mirror the batch-inference AkidaServiceContainer; the
difference is the /shared segment-in / grid-out bus and the raw (detector) reply.
"""
from __future__ import print_function
import array
import json
import mmap
import os
import subprocess
import sys
import threading

import soamapi

DEFAULT_SHM_BYTES = 8 * 1024 * 1024
PIPE_DIR = os.environ.get("AKIDA_PIPELINE_DIR", "/shared/pipeline")


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


class InferenceServiceContainer(soamapi.ServiceContainer):
    def __init__(self):
        super(InferenceServiceContainer, self).__init__()
        self._worker = None
        self._lock = threading.Lock()
        self._shm = None
        self._shm_f = None
        self._shm_path = None
        self._shm_bytes = 0

    def _log(self, msg):
        print("[shard-infer %d] %s" % (os.getpid(), msg), file=sys.stderr)
        sys.stderr.flush()

    def _alloc_shm(self):
        self._shm_bytes = int(os.environ.get("AKIDA_SHM_BYTES", "") or DEFAULT_SHM_BYTES)
        path = "/dev/shm/shard-akida-%d.buf" % os.getpid()
        try:
            f = open(path, "wb+")
            f.truncate(self._shm_bytes)
            mm = mmap.mmap(f.fileno(), self._shm_bytes)
            self._shm_f, self._shm, self._shm_path = f, mm, path
            self._log("shm buffer %s (%d bytes)" % (path, self._shm_bytes))
        except Exception as e:
            self._log("shm alloc failed (%s); inline transport only" % e)
            self._shm = None

    def on_create_service(self, service_context):
        self._alloc_shm()
        py = os.environ["AKIDA_PYTHON"]
        worker = os.environ["AKIDA_WORKER_PY"]
        env = os.environ.copy()
        common = env.get("AKIDA_COMMON_DIR", "/opt/akida-common")
        env["PYTHONPATH"] = env["AKIDA_VENV_SITEPACKAGES"] + os.pathsep + common
        if self._shm is not None:
            env["AKIDA_SHM_PATH"] = self._shm_path
            env["AKIDA_SHM_BYTES"] = str(self._shm_bytes)
        self._log("spawning worker: %s %s" % (py, worker))
        self._worker = subprocess.Popen(
            [py, worker], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=sys.stderr, env=env, bufsize=0)
        ready = self._worker.stdout.readline().decode("utf-8").strip()
        if ready != "READY":
            raise RuntimeError("worker not READY (got %r, rc=%s)"
                               % (ready, self._worker.poll()))
        self._log("worker READY")

    def on_session_enter(self, session_context):
        pass

    def _forward(self, seg_bytes, model):
        """Send one segment tensor to the worker (shm or inline) and return its JSON reply."""
        n = len(seg_bytes)
        header = {"model": model or None, "n": n, "raw": True}
        with self._lock:
            if self._shm is not None and n <= self._shm_bytes:
                self._shm[0:n] = seg_bytes
                self._worker.stdin.write((json.dumps(header) + "\n").encode("utf-8"))
            else:
                header["inline"] = True
                self._worker.stdin.write((json.dumps(header) + "\n").encode("utf-8"))
                self._worker.stdin.write(seg_bytes)
            self._worker.stdin.flush()
            line = self._worker.stdout.readline()
        if not line:
            raise RuntimeError("worker died mid-invoke (rc=%s)" % self._worker.poll())
        return json.loads(line.decode("utf-8").rstrip("\n"))

    def on_invoke(self, task_context):
        in_msg = PipeMessage()
        task_context.populate_task_input(in_msg)
        image_id = str(in_msg.header.get("image_id", ""))
        seg_idx = int(in_msg.header.get("seg_idx", 0))
        model = in_msg.header.get("model")
        d = os.path.join(PIPE_DIR, image_id)
        try:
            with open(os.path.join(d, "seg%d.bin" % seg_idx), "rb") as fh:
                seg_bytes = fh.read()
            r = self._forward(seg_bytes, model)
            if "error" in r:
                reply = {"image_id": image_id, "seg_idx": seg_idx, "ok": False, "error": r["error"]}
            else:
                # persist the raw grid (int32, C-order) on the shared bus for the stitch stage
                with open(os.path.join(d, "grid%d.bin" % seg_idx), "wb") as fh:
                    fh.write(array.array("i", r["output"]).tobytes())
                reply = {"image_id": image_id, "seg_idx": seg_idx, "ok": True,
                         "inference_us": r.get("inference_us"), "host": r.get("host"),
                         "device": r.get("device"), "model": r.get("model")}
        except Exception as e:
            reply = {"image_id": image_id, "seg_idx": seg_idx, "ok": False,
                     "error": "%s: %s" % (type(e).__name__, e)}
        out = soamapi.DefaultTextMessage()
        out.set_text(json.dumps(reply))
        task_context.set_task_output(out)

    def on_session_leave(self):
        pass

    def on_destroy_service(self):
        if self._worker and self._worker.poll() is None:
            try:
                self._worker.stdin.write(b'{"action":"shutdown"}\n')
                self._worker.stdin.flush()
                self._worker.stdin.close()
                self._worker.wait(timeout=5)
            except Exception:
                self._worker.terminate()
        try:
            if self._shm is not None:
                self._shm.close()
            if self._shm_f is not None:
                self._shm_f.close()
            if self._shm_path and os.path.exists(self._shm_path):
                os.unlink(self._shm_path)
        except Exception:
            pass


if __name__ == "__main__":
    InferenceServiceContainer().run()
