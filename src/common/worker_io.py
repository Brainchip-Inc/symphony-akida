"""Framed stdio server for the python3.12 side of a SOAM service (see shard_wire.Py312Worker).

One JSON header line per request, the payload (when there is one) either in the shared
/dev/shm buffer at shm[0:n] or written inline right after the header. One JSON line back.

    {"n": <nbytes>, ...}\n                 -> payload in shm[0:n]
    {"n": <nbytes>, "inline": true, ...}\n + <n bytes>
    {"action": "shutdown"}\n

The handler returns a dict, which is serialised as the reply; raising is fine, the exception
comes back as {"error": ...} and the container turns it into a failed task.
"""
import json
import mmap
import os
import sys


def log(tag, msg):
    sys.stderr.write("[%s] %s\n" % (tag, msg))
    sys.stderr.flush()


def _read_exact(stream, n):
    chunks, got = [], 0
    while got < n:
        block = stream.read(n - got)
        if not block:
            raise EOFError("stdin closed after %d/%d bytes" % (got, n))
        chunks.append(block)
        got += len(block)
    return b"".join(chunks)


def _open_shm(tag):
    path = os.environ.get("AKIDA_SHM_PATH")
    nbytes = int(os.environ.get("AKIDA_SHM_BYTES", "0") or 0)
    if not path or nbytes <= 0:
        return None
    try:
        handle = open(path, "rb")
        mapped = mmap.mmap(handle.fileno(), nbytes, access=mmap.ACCESS_READ)
        log(tag, "shm mapped %s (%d bytes)" % (path, nbytes))
        return mapped
    except Exception as exc:
        log(tag, "shm map failed (%s); inline only" % exc)
        return None


def serve(tag, handler):
    """Announces READY, then runs handler(header, payload) until stdin closes or shutdown."""
    shm = _open_shm(tag)
    stdin = sys.stdin.buffer
    sys.stdout.write("READY\n")
    sys.stdout.flush()

    while True:
        line = stdin.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line.decode("utf-8"))
            if request.get("action") == "shutdown":
                break
            payload = None
            if "n" in request:
                n = int(request["n"])
                payload = (_read_exact(stdin, n) if request.get("inline") or shm is None
                           else shm[0:n])
            response = handler(request, payload)
        except Exception as exc:
            response = {"error": "%s: %s" % (type(exc).__name__, exc)}
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()
