"""Per-node Akida HTTP inference server (Python 3.12) for the serial-http app.

One instance runs on each compute node (started by the container entrypoint when
START_HTTP=1). It owns this node's single Akida chip: it maps a model with
``hw_only=True`` on the real silicon -- whichever family the launcher gave this node,
AKD1500 or AKD1000 -- and serves a small HTTP API that the laptop dashboard round-robins
across the fleet. This is the "before" demo -- HTTP + serial round-robin, one chip busy
at a time -- in contrast to the batch-inference app's concurrent SOAM fan-out.

The on-chip logic (device select, hw_only map, forward) is the SHARED ``akida_chip``
module (baked at /opt/akida-common), the very same code the SOAM worker uses. Here we
load with allow_software=True so a model that will not map hw_only still runs (in
software) and the node honestly reports SOFTWARE instead of ON-CHIP.

HTTP API (matches the dashboard's akida_client.py contract):
  GET  /health              -> {host, chip_node, model, akida_mapped, hardware_present,
                                device, product, map_error, akida_version}
  GET  /models              -> {models:[{name,input_shape,num_classes,class_names,size_bytes}], current}
  POST /load    {name}      -> {model, akida_mapped, device, product, ...} (maps hw_only on-chip)
  POST /reload  {name}      -> same as /load
  POST /unload              -> {ok:true}
  POST /infer   {input:[…]} -> {cls, cls_name, inference_us, host, device, product, hardware, mode}

`device` is the chip's own desc ("PCIe/AKD1500/16MB/0"), `product` the name to show
("AKD1500"), and `chip_node` the physical /dev node this container was pinned to
("akd1500_3"). The last one is the only per-node discriminator: the entrypoint exposes
every node's chip as slot 0, so `device` is byte-identical across the fleet.
"""
import json
import os
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Shared code baked at /opt/akida-common (also set on PYTHONPATH by the wrapper).
sys.path.insert(0, os.environ.get("AKIDA_COMMON_DIR", "/opt/akida-common"))
from akida_chip import Chip, select_device, akida_version, MODELS_DIR  # noqa: E402
import models as allowlist  # noqa: E402  shared classifier-model allowlist

HOST = socket.gethostname()
# The /dev node up.sh pinned to this container ("akd1500_3"). Set by docker run and still in
# our env because the entrypoint starts us with `su egoadmin -c` (no `-`, so the environment
# survives). It is the only thing that tells two nodes apart -- see the module docstring.
CHIP_NODE = os.environ.get("AKIDA_CHIP_NODE", "").strip() or None
PORT = int(os.environ.get("HTTP_PORT", "8790"))
DEFAULT_MODEL = os.environ.get("AKIDA_DEFAULT_MODEL", "").strip()

_lock = threading.Lock()   # the chip serialises work; one task in flight per node
_chip = None               # Chip once a device is acquired, else None
_hw_present = False


def _init():
    """Acquire this node's chip and map the default model (software fallback allowed)."""
    global _chip, _hw_present
    try:
        dev = select_device()
    except Exception as e:
        sys.stderr.write("[http-server] no Akida device on this node: %s\n" % e)
        sys.stderr.flush()
        _hw_present = False
        _chip = None
        return
    _hw_present = True
    _chip = Chip(dev)
    if DEFAULT_MODEL:
        try:
            _chip.load(DEFAULT_MODEL, allow_software=True)
        except Exception as e:
            sys.stderr.write("[http-server] default model %r failed: %s\n" % (DEFAULT_MODEL, e))
            sys.stderr.flush()


def _model_list():
    """Allowlisted models present in MODELS_DIR, enriched from the meta sidecar."""
    out = []
    for stem in allowlist.visible(MODELS_DIR):
        path = os.path.join(MODELS_DIR, stem + ".fbz")
        info = {"name": stem,
                "size_bytes": os.path.getsize(path) if os.path.isfile(path) else None}
        try:
            meta = json.load(open(os.path.join(MODELS_DIR, stem + "_meta.json")))
            info["input_shape"] = meta.get("input_shape")
            info["num_classes"] = meta.get("num_classes")
            info["class_names"] = meta.get("class_names")
        except Exception:
            pass
        out.append(info)
    return out


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass  # keep the per-node log quiet; the wrapper logs startup

    def _send(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self):
        n = int(self.headers.get("Content-Length", 0) or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode("utf-8"))
        except Exception:
            return {}

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {
                "host": HOST,
                "chip_node": CHIP_NODE,
                "model": _chip.stem if _chip else None,
                "akida_mapped": bool(_chip and _chip.on_chip),
                "hardware_present": _hw_present,
                "device": _chip.desc if _chip else None,
                "product": _chip.product if _chip else None,
                "map_error": _chip.map_error if _chip else None,
                "akida_version": akida_version(),
            })
        elif self.path == "/models":
            cur = {"name": _chip.stem} if (_chip and _chip.stem) else None
            self._send(200, {"models": _model_list(), "current": cur})
        else:
            self._send(404, {"error": "not found: %s" % self.path})

    def do_POST(self):
        body = self._read_body()
        try:
            if self.path in ("/load", "/reload"):
                name = body.get("name")
                if not name:
                    return self._send(400, {"error": "no model name"})
                if _chip is None:
                    return self._send(503, {"error": "no akida device on this node"})
                with _lock:
                    info = _chip.load(name, allow_software=True)
                self._send(200, dict(info, model=info["name"]))
            elif self.path == "/unload":
                if _chip is not None:
                    with _lock:
                        _chip.unload()
                self._send(200, {"ok": True})
            elif self.path == "/infer":
                if _chip is None:
                    return self._send(503, {"error": "no akida device on this node"})
                vals = body.get("input")
                if vals is None:
                    return self._send(400, {"error": "no input"})
                with _lock:
                    r = _chip.infer(vals)
                    on_chip = _chip.on_chip
                # The chip's own product name, never a literal: this fleet is AKD1500 and
                # used to report AKD1000. Guaranteed non-empty (akida_product), which the
                # dashboard's "all N ON-CHIP" tally depends on -- it counts truthy values.
                r["hardware"] = _chip.product if on_chip else None
                r["mode"] = "on-chip" if on_chip else "software"
                self._send(200, r)
            else:
                self._send(404, {"error": "not found: %s" % self.path})
        except Exception as e:
            self._send(500, {"error": "%s: %s" % (type(e).__name__, e)})


def main():
    _init()
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    # "listening on" is up.sh's readiness marker for this app -- keep the substring.
    sys.stderr.write("[http-server] %s listening on :%d  chip=%s product=%s hw_present=%s"
                     " model=%s on_chip=%s\n"
                     % (HOST, PORT, CHIP_NODE,
                        _chip.product if _chip else None, _hw_present,
                        _chip.stem if _chip else None,
                        _chip.on_chip if _chip else False))
    sys.stderr.flush()
    srv.serve_forever()


if __name__ == "__main__":
    main()
