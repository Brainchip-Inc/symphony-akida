"""SOAM ServiceContainer (Python 3.6) for the Akida on-chip service.

Symphony CE 7.3.2's soamapi Python binding is compiled for the 3.6 ABI, while
akida needs 3.12 -- so the actual inference runs in a 3.12 subprocess
(akida_worker.py) that this container talks to over stdio (JSON per line).

- on_create_service: spawn the worker and require READY. If the worker exits
  first (no device, or the default model would not map hw_only), this raises
  and the SI does not come up -- the node is not used.
- on_invoke: forward the task's JSON to the worker, return the worker's reply.
"""
from __future__ import print_function
import json
import os
import subprocess
import sys
import threading

import soamapi


class AkidaServiceContainer(soamapi.ServiceContainer):
    def __init__(self):
        super(AkidaServiceContainer, self).__init__()
        self._worker = None
        self._lock = threading.Lock()

    def _log(self, msg):
        print("[akida-svc %d] %s" % (os.getpid(), msg), file=sys.stderr)
        sys.stderr.flush()

    def on_create_service(self, service_context):
        py = os.environ["AKIDA_PYTHON"]
        worker = os.environ["AKIDA_WORKER_PY"]
        env = os.environ.copy()
        env["PYTHONPATH"] = env["AKIDA_VENV_SITEPACKAGES"]
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
        in_msg = soamapi.DefaultTextMessage()
        task_context.populate_task_input(in_msg)
        payload = in_msg.get_text()
        with self._lock:
            self._worker.stdin.write((payload + "\n").encode("utf-8"))
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


if __name__ == "__main__":
    AkidaServiceContainer().run()
