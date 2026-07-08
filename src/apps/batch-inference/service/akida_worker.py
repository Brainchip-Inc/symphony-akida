"""Akida on-chip inference worker (Python 3.12).

Runs as a subprocess of the SOAM ServiceContainer and speaks JSON-per-line over
stdio. It acquires a real Akida device, maps a model onto it with
``hw_only=True``, and runs ``forward()`` on-chip. Strict rules:

- No device visible            -> exit without READY (the SI never serves; the node is not used).
- Model will not map hw_only   -> hard error (that model does not run on this silicon).
- Ready only after a successful hardware map; each input is then a plain forward.

Protocol (one JSON object per line):
  {"model":"kws_keyword_spotting","input":[...]} -> {cls,cls_name,inference_us,host,device,model}
  {"action":"shutdown"}                          -> exit
The model is (re)mapped only when it differs from the one currently on the chip,
so a batch of one model maps once and then just forwards.
"""
import json
import os
import socket
import sys
import time

import numpy as np
import akida

HOST = socket.gethostname()
MODELS_DIR = os.environ.get("AKIDA_MODELS_DIR", "/shared/models")


def _fail(msg):
    sys.stderr.write("[akida-worker] FATAL: %s\n" % msg)
    sys.stderr.flush()
    sys.exit(1)


def select_device():
    """Return this node's Akida device, chosen by a stable order + index."""
    devs = akida.devices()
    if not devs:
        _fail("no Akida device present; node cannot run on-chip inference")
    devs = sorted(devs, key=lambda d: str(getattr(d, "desc", d)))
    idx = int(os.environ.get("AKIDA_DEVICE_INDEX", "0"))
    if idx >= len(devs):
        _fail("AKIDA_DEVICE_INDEX=%d but only %d device(s) visible" % (idx, len(devs)))
    dev = devs[idx]
    sys.stderr.write("[akida-worker] %s -> device[%d] %s\n"
                     % (HOST, idx, getattr(dev, "desc", str(dev))))
    sys.stderr.flush()
    return dev


def _stem(name):
    name = os.path.basename(name)
    return name[:-4] if name.endswith(".fbz") else name


def _classes(path, n):
    base = path[:-4]
    for suf in ("_meta.json", "_params.json", ".classes.json", ".json"):
        p = base + suf
        if not os.path.isfile(p):
            continue
        try:
            meta = json.load(open(p))
        except Exception:
            continue
        for k in ("class_names", "class_labels", "classes", "labels"):
            v = meta.get(k)
            if isinstance(v, list) and len(v) == n:
                return [str(x) for x in v]
    return [str(i) for i in range(n)]


class Chip:
    def __init__(self, device):
        self.device = device
        self.desc = str(getattr(device, "desc", device))
        self.model = None
        self.stem = None
        self.ishape = None
        self.classes = None

    def load(self, name):
        path = name if os.path.isabs(name) else os.path.join(MODELS_DIR, name)
        if not path.endswith(".fbz"):
            path += ".fbz"
        if not os.path.isfile(path):
            raise FileNotFoundError("no such model: %s" % path)
        m = akida.Model(path)
        # AllNps spreads the model across all neural processors on the device.
        m.map(self.device, hw_only=True, mode=akida.MapMode.AllNps)
        self.model = m
        self.stem = _stem(path)
        self.ishape = tuple(int(d) for d in m.input_shape)
        nout = int(np.prod(m.output_shape))
        self.classes = _classes(path, nout)
        return {"name": self.stem, "input_shape": list(self.ishape),
                "num_classes": nout, "class_names": self.classes, "device": self.desc}

    def infer(self, raw):
        if self.model is None:
            raise RuntimeError("no model mapped")
        arr = np.asarray(raw)
        n = int(np.prod(self.ishape))
        if arr.size != n:
            raise ValueError("input has %d values, model expects %d %s"
                             % (arr.size, n, list(self.ishape)))
        x = arr.reshape((1,) + tuple(self.ishape)).astype(np.uint8)
        t0 = time.perf_counter()
        y = self.model.forward(x)
        us = int((time.perf_counter() - t0) * 1e6)
        logits = np.asarray(y).squeeze().astype(int).tolist()
        if isinstance(logits, int):
            logits = [logits]
        cls = int(np.argmax(logits))
        return {"cls": cls,
                "cls_name": self.classes[cls] if cls < len(self.classes) else str(cls),
                "inference_us": us, "host": HOST, "device": self.desc, "model": self.stem}


def main():
    # Strict readiness: acquire the device (hard-fail exits before READY), then
    # map the default model on-chip. READY is emitted only after a successful
    # hw_only map -- no device or a failed map means no READY, so the SOAM
    # instance never serves and the node is not used. Each compute node is
    # pinned to a single chip (see the entrypoint), so this enumerate+map is
    # fast and does not trip the SIM's create deadline.
    chip = Chip(select_device())
    default = os.environ.get("AKIDA_DEFAULT_MODEL", "").strip()
    if default:
        try:
            chip.load(default)
            sys.stderr.write("[akida-worker] mapped default %s hw_only\n" % default)
        except Exception as e:
            _fail("default model %r failed to map hw_only: %s" % (default, e))
    sys.stderr.flush()

    sys.stdout.write("READY\n")
    sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            if req.get("action") == "shutdown":
                break
            if "input" in req:
                model = req.get("model")
                if model and _stem(model) != chip.stem:
                    chip.load(model)
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
