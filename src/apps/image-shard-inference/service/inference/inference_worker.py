"""Akida on-chip inference worker (Python 3.12) for the shard pipeline's INFERENCE stage.

Runs as a subprocess of InferenceServiceContainer and speaks the same framed stdio protocol
as the batch-inference worker: a JSON header line per task, the tensor either in the shared
/dev/shm buffer (shm[0:n]) or inline. The only difference is the reply -- this stage runs a
DETECTOR, so it returns the RAW output grid (via the shared Chip.forward_raw) rather than an
argmax class:

  {"model": <str|null>, "n": <nbytes>, "raw": true}\\n            -> tensor in shm[0:n]
  {"model": <str|null>, "n": <nbytes>, "raw": true, "inline": true}\\n + <n bytes>
  {"action": "shutdown"}\\n

Reply (one JSON line): {output:[ints], output_shape:[...], inference_us, host, device, model}.
Device select + hw_only map + forward all live in the shared akida_chip core (/opt/akida-common),
identical to the batch worker; this file is just the SOAM stdio transport around it.
"""
import json
import mmap
import os
import sys
import time

import numpy as np

from akida_chip import Chip, select_device, _stem  # shared on-chip core (/opt/akida-common)


def _fail(msg):
    sys.stderr.write("[shard-infer-worker] FATAL: %s\n" % msg)
    sys.stderr.flush()
    sys.exit(1)


def _read_exact(stream, n):
    chunks, got = [], 0
    while got < n:
        b = stream.read(n - got)
        if not b:
            raise EOFError("stdin closed after %d/%d bytes" % (got, n))
        chunks.append(b)
        got += len(b)
    return b"".join(chunks)


def _open_shm():
    path = os.environ.get("AKIDA_SHM_PATH")
    nbytes = int(os.environ.get("AKIDA_SHM_BYTES", "0") or 0)
    if not path or nbytes <= 0:
        return None
    try:
        f = open(path, "rb")
        mm = mmap.mmap(f.fileno(), nbytes, access=mmap.ACCESS_READ)
        sys.stderr.write("[shard-infer-worker] shm mapped %s (%d bytes)\n" % (path, nbytes))
        sys.stderr.flush()
        return mm
    except Exception as e:
        sys.stderr.write("[shard-infer-worker] shm map failed (%s); inline only\n" % e)
        sys.stderr.flush()
        return None


def main():
    # Retry device acquisition: on an SI restart the just-killed worker may still hold the
    # chip's driver lock for a moment. Retrying a few seconds lets it release before we FATAL,
    # turning a would-be restart cascade into a brief hiccup.
    device = None
    for attempt in range(12):
        try:
            device = select_device()
            break
        except Exception as e:
            sys.stderr.write("[shard-infer-worker] device not ready (attempt %d/12): %s\n"
                             % (attempt + 1, e))
            sys.stderr.flush()
            time.sleep(2)
    if device is None:
        _fail("no Akida device after retries; node cannot run on-chip inference")
    chip = Chip(device)
    default = os.environ.get("AKIDA_DEFAULT_MODEL", "").strip()
    if default:
        try:
            chip.load(default)  # strict hw_only: raises if the model will not map on-chip
            sys.stderr.write("[shard-infer-worker] mapped default %s hw_only\n" % default)
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
            break
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
                    arr = np.frombuffer(_read_exact(stdin, n), dtype=np.uint8, count=n)
                else:
                    arr = np.frombuffer(shm, dtype=np.uint8, count=n)
                resp = chip.forward_raw(arr) if req.get("raw") else chip.infer(arr)
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
