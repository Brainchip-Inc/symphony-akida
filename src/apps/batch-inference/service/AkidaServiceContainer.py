"""SOAM ServiceContainer (Python 3.6) for the Akida on-chip service.

Symphony CE 7.3.2's soamapi Python binding is compiled for the 3.6 ABI, while
akida needs 3.12 -- so the actual inference runs in a 3.12 subprocess
(akida_worker.py) that this container talks to over stdio.

The input tensor arrives from the client as raw bytes (a binary soamapi Message,
not a JSON int array) and is handed to the worker through a /dev/shm shared buffer
we allocate here; only a tiny JSON header goes over the pipe. This avoids the
~4x ASCII blow-up and the per-task JSON parse of the old int-array payload.

- on_create_service: allocate + mmap the shm buffer, spawn the worker (passing the
  buffer path/size), and require READY. If the worker exits first (no device, or
  the default model would not map hw_only), this raises and the SI does not come
  up -- the node is not used.
- on_invoke: write the tensor into the shm buffer, send a {model,n} header, return
  the worker's JSON reply.
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

# Default shared-buffer size (8 MiB) if the env does not override it. Sized to
# hold one input tensor; bigger future models raise AKIDA_SHM_BYTES (+ --shm-size).
DEFAULT_SHM_BYTES = 8 * 1024 * 1024


class TensorInputMessage(soamapi.Message):
    """Binary task input. Wire format MUST match soam_client.py:
       write_string(model); write_byte_array(array('B', tensor_bytes), 0, len)."""

    def __init__(self):
        super(TensorInputMessage, self).__init__()
        self.model = ""
        self.data = b""  # raw uint8 tensor bytes

    def on_serialize(self, stream):
        arr = array.array("B", self.data)
        stream.write_string(self.model or "")
        stream.write_byte_array(arr, 0, len(arr))

    def on_deserialize(self, stream):
        self.model = stream.read_string()
        self.data = stream.read_byte_array("B").tobytes()


class AkidaServiceContainer(soamapi.ServiceContainer):
    def __init__(self):
        super(AkidaServiceContainer, self).__init__()
        self._worker = None
        self._lock = threading.Lock()
        self._shm = None
        self._shm_f = None
        self._shm_path = None
        self._shm_bytes = 0

    def _log(self, msg):
        print("[akida-svc %d] %s" % (os.getpid(), msg), file=sys.stderr)
        sys.stderr.flush()

    def _alloc_shm(self):
        """Create + mmap a /dev/shm buffer the worker will read. Best-effort:
        on failure we fall back to sending tensors inline over the pipe."""
        self._shm_bytes = int(os.environ.get("AKIDA_SHM_BYTES", "") or DEFAULT_SHM_BYTES)
        path = "/dev/shm/akida-%d.buf" % os.getpid()
        try:
            f = open(path, "wb+")
            f.truncate(self._shm_bytes)
            mm = mmap.mmap(f.fileno(), self._shm_bytes)  # MAP_SHARED, read+write
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
        env["PYTHONPATH"] = env["AKIDA_VENV_SITEPACKAGES"]
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

    def on_invoke(self, task_context):
        in_msg = TensorInputMessage()
        task_context.populate_task_input(in_msg)
        data = in_msg.data
        n = len(data)
        header = {"model": in_msg.model or None, "n": n}
        with self._lock:
            if self._shm is not None and n <= self._shm_bytes:
                self._shm[0:n] = data            # write tensor into shared buffer
                self._worker.stdin.write((json.dumps(header) + "\n").encode("utf-8"))
            else:                                # buffer missing or too small: inline
                header["inline"] = True
                self._worker.stdin.write((json.dumps(header) + "\n").encode("utf-8"))
                self._worker.stdin.write(data)
            self._worker.stdin.flush()
            line = self._worker.stdout.readline()
        if not line:
            raise RuntimeError("worker died mid-invoke (rc=%s)" % self._worker.poll())
        out_msg = soamapi.DefaultTextMessage()
        out_msg.set_text(line.decode("utf-8").rstrip("\n"))
        task_context.set_task_output(out_msg)

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
    AkidaServiceContainer().run()
