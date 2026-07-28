"""Shared plumbing for the shard pipeline's SOAM side (Python 3.6).

Two pieces, both used by all three service containers and (PipeMessage only) by the client:

  PipeMessage  the task wire format. Big tensors travel over /shared; only tiny references,
               the input image and the final detections cross SOAM.
  Py312Worker  a python3.12 subprocess behind a framed stdio protocol, with a /dev/shm buffer
               for the payload. soamapi is python3.6-only and numpy/akida are python3.12-only,
               so every stage that needs real array math runs it out of process. The inference
               stage has always worked this way; segment and stitch do too, because the tile
               downscale and the detection merge are numpy, and hand-porting the merge to
               python3.6 list math is exactly the rewrite that quietly loses the truncated
               flag (worth ~5 mAP50 and 8.9 recall points).
"""
from __future__ import print_function
import array
import json
import mmap
import os
import subprocess
import sys

import soamapi

DEFAULT_SHM_BYTES = 8 * 1024 * 1024


class PipeMessage(soamapi.Message):
    """Shard pipeline wire format (MUST match every container and the client):
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


class Py312Worker(object):
    """A python3.12 worker subprocess: one JSON request line in, one JSON reply line out.

    The payload, when there is one, goes through a shared /dev/shm buffer rather than the
    pipe; the request carries its length and the worker reads shm[0:n]. Oversized payloads
    (or a failed shm allocation) fall back to writing the bytes inline after the header.
    """

    def __init__(self, tag):
        self.tag = tag
        self._proc = None
        self._shm = None
        self._shm_file = None
        self._shm_path = None
        self._shm_bytes = 0

    def log(self, msg):
        print("[%s %d] %s" % (self.tag, os.getpid(), msg), file=sys.stderr)
        sys.stderr.flush()

    def start(self, want_shm=True):
        if want_shm:
            self._alloc_shm()
        python = os.environ["AKIDA_PYTHON"]
        script = os.environ["AKIDA_WORKER_PY"]
        env = os.environ.copy()
        common = env.get("AKIDA_COMMON_DIR", "/opt/akida-common")
        env["PYTHONPATH"] = env["AKIDA_VENV_SITEPACKAGES"] + os.pathsep + common
        if self._shm is not None:
            env["AKIDA_SHM_PATH"] = self._shm_path
            env["AKIDA_SHM_BYTES"] = str(self._shm_bytes)
        self.log("spawning worker: %s %s" % (python, script))
        self._proc = subprocess.Popen([python, script], stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE, stderr=sys.stderr, env=env,
                                      bufsize=0)
        ready = self._proc.stdout.readline().decode("utf-8").strip()
        if ready != "READY":
            raise RuntimeError("worker not READY (got %r, rc=%s)" % (ready, self._proc.poll()))
        self.log("worker READY")

    def _alloc_shm(self):
        self._shm_bytes = int(os.environ.get("AKIDA_SHM_BYTES", "") or DEFAULT_SHM_BYTES)
        path = "/dev/shm/%s-%d.buf" % (self.tag, os.getpid())
        try:
            handle = open(path, "wb+")
            handle.truncate(self._shm_bytes)
            self._shm_file = handle
            self._shm = mmap.mmap(handle.fileno(), self._shm_bytes)
            self._shm_path = path
            self.log("shm buffer %s (%d bytes)" % (path, self._shm_bytes))
        except Exception as exc:
            self.log("shm alloc failed (%s); inline transport only" % exc)
            self._shm = None

    def request(self, header, payload=None):
        """Sends one request (payload via shm when it fits) and returns the parsed reply."""
        header = dict(header)
        if payload is not None:
            header["n"] = len(payload)
            if self._shm is None or len(payload) > self._shm_bytes:
                header["inline"] = True
            else:
                self._shm[0:len(payload)] = payload
        self._proc.stdin.write((json.dumps(header) + "\n").encode("utf-8"))
        if payload is not None and header.get("inline"):
            self._proc.stdin.write(payload)
        self._proc.stdin.flush()
        line = self._proc.stdout.readline()
        if not line:
            raise RuntimeError("worker died mid-invoke (rc=%s)" % self._proc.poll())
        return json.loads(line.decode("utf-8").rstrip("\n"))

    def stop(self):
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.stdin.write(b'{"action":"shutdown"}\n')
                self._proc.stdin.flush()
                self._proc.stdin.close()
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.terminate()
        try:
            if self._shm is not None:
                self._shm.close()
            if self._shm_file is not None:
                self._shm_file.close()
            if self._shm_path and os.path.exists(self._shm_path):
                os.unlink(self._shm_path)
        except Exception:
            pass
