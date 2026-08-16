"""Shared Akida on-chip core: device selection + hw_only model map + forward.

This is the single source of the on-chip logic used by BOTH demo apps:

  * batch-inference : the SOAM worker (akida_worker.py) imports Chip/select_device
                      and runs strictly (hw_only map or the node is not used).
  * serial-http-round-robin : the per-node HTTP server (http_server.py) imports the
                      same Chip but allows a software fallback so it can display an
                      honest ON-CHIP vs SOFTWARE badge.

Baked into the image at /opt/akida-common and put on the in-container PYTHONPATH by
both service wrappers. Requires the akida venv (numpy + akida) on the path -- it is a
Python 3.12 module and is never imported by host-side tooling.
"""
import json
import os
import socket
import sys
import time

import numpy as np
import akida

import akida_product

HOST = socket.gethostname()
MODELS_DIR = os.environ.get("AKIDA_MODELS_DIR", "/shared/models")


def akida_version():
    return str(getattr(akida, "__version__", "?"))


def device_product(device):
    """This device's product name -- "AKD1500", "AKD1000". Never empty.

    Keyed on the HwVersion, which is exact, falling back to parsing the desc. Both tables
    live in akida_product so the host-side dashboards (no akida, no chip) name the silicon
    the same way this does.
    """
    return (akida_product.from_version(getattr(device, "version", None))
            or akida_product.from_desc(getattr(device, "desc", device)))


def select_device():
    """Return this node's Akida device, chosen by a stable order + index.

    Raises RuntimeError if no device is visible or the index is out of range. Each
    node is pinned to a single chip (exposed as index 0) by the container entrypoint.
    """
    devs = akida.devices()
    if not devs:
        raise RuntimeError("no Akida device present; node cannot run on-chip inference")
    devs = sorted(devs, key=lambda d: str(getattr(d, "desc", d)))
    idx = int(os.environ.get("AKIDA_DEVICE_INDEX", "0"))
    if idx >= len(devs):
        raise RuntimeError("AKIDA_DEVICE_INDEX=%d but only %d device(s) visible"
                           % (idx, len(devs)))
    dev = devs[idx]
    sys.stderr.write("[akida-chip] %s -> device[%d] %s (%s)\n"
                     % (HOST, idx, getattr(dev, "desc", str(dev)), device_product(dev)))
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


def _map_mode(path):
    """Per-model MapMode name from the meta sidecar (default AllNps).

    Read from each model's meta ("map_mode"). The demo models use "HwPr" (hardware
    pipeline), verified to map hw_only and run cleanly (incl. sparse/all-zero inputs)
    on AKD1500. Other options: "AllNps" spreads each layer across all NPs for max
    parallelism, but for a high-sparsity model some NP partitions can receive all-zero
    activity on certain inputs, which the hardware's output accounting mishandles ->
    a 5s fetch-timeout; "Minimal" keeps the mapping compact and avoids that. Pick
    whichever a given model needs in its meta.
    """
    base = path[:-4]
    for suf in ("_meta.json", "_params.json", ".json"):
        p = base + suf
        if not os.path.isfile(p):
            continue
        try:
            return json.load(open(p)).get("map_mode") or "AllNps"
        except Exception:
            break
    return "AllNps"


class Chip:
    """One Akida device with at most one model mapped onto it.

    load() maps a model hw_only=True. In strict mode (allow_software=False, the SOAM
    worker's contract) a map failure raises. With allow_software=True (the HTTP
    server) a map failure leaves the model unmapped so forward() runs on the software
    backend, and self.on_chip records which happened -- driving the ON-CHIP/SOFTWARE
    badge. Either way infer() calls model.forward() and returns the predicted class.
    """

    def __init__(self, device):
        self.device = device
        self.desc = str(getattr(device, "desc", device))
        # The product name every app shows. desc is the silicon name ("NSoC_v2") and, since
        # the entrypoint pins one chip per container at slot 0, it is also identical on every
        # node -- so it is what the chip IS, not which chip it is.
        self.product = device_product(device)
        self.model = None
        self.stem = None
        self.ishape = None
        self.n = None
        self.classes = None
        self.on_chip = False
        self.map_error = None

    def load(self, name, allow_software=False):
        path = name if os.path.isabs(name) else os.path.join(MODELS_DIR, name)
        if not path.endswith(".fbz"):
            path += ".fbz"
        if not os.path.isfile(path):
            raise FileNotFoundError("no such model: %s" % path)
        m = akida.Model(path)
        mode_name = _map_mode(path)
        mode = getattr(akida.MapMode, mode_name, akida.MapMode.AllNps)
        try:
            m.map(self.device, hw_only=True, mode=mode)
            self.on_chip = True
            self.map_error = None
            sys.stderr.write("[akida-chip] mapped %s mode=%s hw_only (on-chip)\n"
                             % (_stem(path), mode_name))
        except Exception as e:
            if not allow_software:
                raise
            self.on_chip = False
            self.map_error = str(e)
            sys.stderr.write("[akida-chip] %s did not map hw_only (%s); running in software\n"
                             % (_stem(path), e))
        sys.stderr.flush()
        self.model = m
        self.stem = _stem(path)
        self.ishape = tuple(int(d) for d in m.input_shape)
        self.n = int(np.prod(self.ishape))
        nout = int(np.prod(m.output_shape))
        self.classes = _classes(path, nout)
        return {"name": self.stem, "input_shape": list(self.ishape),
                "num_classes": nout, "class_names": self.classes, "device": self.desc,
                "product": self.product,
                "akida_mapped": self.on_chip, "map_error": self.map_error}

    def unload(self):
        self.model = None
        self.stem = None
        self.on_chip = False
        self.map_error = None

    def infer(self, arr):
        """arr: 1-D uint8 ndarray/buffer of exactly prod(input_shape) values."""
        if self.model is None:
            raise RuntimeError("no model mapped")
        arr = np.asarray(arr, dtype=np.uint8).reshape(-1)
        if arr.size != self.n:
            raise ValueError("input has %d values, model expects %d %s"
                             % (arr.size, self.n, list(self.ishape)))
        x = arr.reshape((1,) + tuple(self.ishape))
        t0 = time.perf_counter()
        y = self.model.forward(x)
        us = int((time.perf_counter() - t0) * 1e6)
        logits = np.asarray(y).squeeze().astype(int).tolist()
        if isinstance(logits, int):
            logits = [logits]
        cls = int(np.argmax(logits))
        return {"cls": cls,
                "cls_name": self.classes[cls] if cls < len(self.classes) else str(cls),
                "inference_us": us, "host": HOST, "device": self.desc,
                "product": self.product, "model": self.stem}

    def predict_tile(self, arr):
        """Like infer(), but return the model's float output tensor instead of an argmax class.

        A detector has no single "class" -- the meaningful result is the whole output grid,
        decoded downstream by the caller. Returns (output, inference_us, identity dict).

        Uses model.predict(), NOT model.forward(). predict rescales the integer potentials
        back into the float range the decode expects; forward returns raw int32. The rescale
        is per output channel -- on this detector the 125 channel scales span 554 to 28,029 --
        so no single global constant can stand in for it, and the two calls cost the same.
        """
        if self.model is None:
            raise RuntimeError("no model mapped")
        arr = np.asarray(arr, dtype=np.uint8).reshape(-1)
        if arr.size != self.n:
            raise ValueError("input has %d values, model expects %d %s"
                             % (arr.size, self.n, list(self.ishape)))
        x = arr.reshape((1,) + tuple(self.ishape))
        t0 = time.perf_counter()
        y = self.model.predict(x)
        us = int((time.perf_counter() - t0) * 1e6)
        return (np.asarray(y)[0], us,
                {"host": HOST, "device": self.desc, "product": self.product,
                 "model": self.stem})
