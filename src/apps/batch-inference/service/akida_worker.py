"""Akida on-chip inference worker (Python 3.12).

Runs as a subprocess of the SOAM ServiceContainer and speaks a small framed
protocol over stdio. It acquires a real Akida device, maps a model onto it with
``hw_only=True``, and runs ``forward()`` on-chip. Strict rules:

- No device visible            -> exit without READY (the SI never serves; the node is not used).
- Model will not map hw_only   -> hard error (that model does not run on this silicon).
- Ready only after a successful hardware map; each input is then a plain forward.

The device selection + hw_only map + forward live in the shared ``akida_chip``
module (baked at /opt/akida-common, shared with the serial-http-round-robin app);
this file is just the SOAM stdio transport around it.

Transport. The input tensor is raw uint8 bytes (not a JSON int array). Each task is
a JSON **header line** followed, for large tensors, by the bytes themselves:

  {"model": <str|null>, "n": <nbytes>}\\n                 -> tensor is in the shared buffer, shm[0:n]
  {"model": <str|null>, "n": <nbytes>, "inline": true}\\n + <n raw bytes>   -> tensor follows on stdin
  {"model": <str|null>, "input": [ints...]}\\n            -> back-compat JSON array
  {"action": "shutdown"}\\n                               -> exit
  {"action": "load", "name": <model>}\\n                  -> (re)map a model

The shared buffer is a /dev/shm file the ServiceContainer writes and we mmap once
at startup (AKIDA_SHM_PATH / AKIDA_SHM_BYTES); reading it is zero-copy. The SI
serialises invokes, so shm[0:n] is stable between our read and our reply.

Reply is one JSON line: {cls, cls_name, inference_us, host, device, model}. The
model is (re)mapped only when it differs from the one on the chip, so a batch of
one model maps once and then just forwards.
"""
import json
import mmap
import os
import sys

import numpy as np

from akida_chip import Chip, select_device, _stem  # shared on-chip core (/opt/akida-common)


def _fail(msg):
    sys.stderr.write("[akida-worker] FATAL: %s\n" % msg)
    sys.stderr.flush()
    sys.exit(1)


def _read_exact(stream, n):
    """Read exactly n bytes from a binary stream (pipes can short-read)."""
    chunks = []
    got = 0
    while got < n:
        b = stream.read(n - got)
        if not b:
            raise EOFError("stdin closed after %d/%d bytes" % (got, n))
        chunks.append(b)
        got += len(b)
    return b"".join(chunks)


def _open_shm():
    """mmap the shared /dev/shm buffer the ServiceContainer writes, if configured."""
    path = os.environ.get("AKIDA_SHM_PATH")
    nbytes = int(os.environ.get("AKIDA_SHM_BYTES", "0") or 0)
    if not path or nbytes <= 0:
        return None
    try:
        f = open(path, "rb")  # SI creates + sizes it before spawning us
        mm = mmap.mmap(f.fileno(), nbytes, access=mmap.ACCESS_READ)
        sys.stderr.write("[akida-worker] shm mapped %s (%d bytes)\n" % (path, nbytes))
        sys.stderr.flush()
        return mm
    except Exception as e:
        sys.stderr.write("[akida-worker] shm map failed (%s); inline only\n" % e)
        sys.stderr.flush()
        return None


def main():
    # Strict readiness: acquire the device (hard-fail exits before READY), then
    # map the default model on-chip. READY is emitted only after a successful
    # hw_only map -- no device or a failed map means no READY, so the SOAM
    # instance never serves and the node is not used. Each compute node is
    # pinned to a single chip (see the entrypoint), so this enumerate+map is
    # fast and does not trip the SIM's create deadline.
    try:
        device = select_device()
    except Exception as e:
        _fail(str(e))
    chip = Chip(device)
    default = os.environ.get("AKIDA_DEFAULT_MODEL", "").strip()
    if default:
        try:
            chip.load(default)  # strict: allow_software defaults False -> raises if not hw_only
            sys.stderr.write("[akida-worker] mapped default %s hw_only\n" % default)
        except Exception as e:
            _fail("default model %r failed to map hw_only: %s" % (default, e))
    shm = _open_shm()
    sys.stderr.flush()

    stdin = sys.stdin.buffer
    sys.stdout.write("READY\n")
    sys.stdout.flush()

    while True:
        header = stdin.readline()
        if not header:
            break  # SI closed the pipe
        header = header.strip()
        if not header:
            continue
        try:
            req = json.loads(header.decode("utf-8"))
            if req.get("action") == "shutdown":
                break
            model = req.get("model")
            if model and _stem(model) != chip.stem:
                chip.load(model)
            if "n" in req:
                n = int(req["n"])
                if req.get("inline") or shm is None:
                    buf = _read_exact(stdin, n)
                    arr = np.frombuffer(buf, dtype=np.uint8, count=n)
                else:
                    arr = np.frombuffer(shm, dtype=np.uint8, count=n)
                resp = chip.infer(arr)
            elif "input" in req:  # back-compat JSON int array
                resp = chip.infer(req["input"])
            elif req.get("action") == "load":
                resp = {"ok": True, "model": chip.load(req["name"])}
            else:
                resp = {"error": "unknown request", "got": req}
        except Exception as e:
            resp = {"error": "%s: %s" % (type(e).__name__, e)}
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
