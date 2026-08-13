"""Python client for the Akida model service (runs on the laptop/host).

The service runs as a SOAM SI on each Symphony compute node and exposes an
HTTP API on a per-node port: `up.sh` publishes compute-j on host port 8790+j,
so compute-0 -> localhost:8790, compute-1 -> 8791, and so on (override the
base with AKIDA_PORT_BASE). This client talks to one node's endpoint for
load/unload/reload/infer and can stage a local `.fbz` into the cluster's
shared models directory so the worker can load it.

`up.sh` already seeds every model in `models/` into that shared dir, so
staging is only needed for an `.fbz` that does not live in the repo.

Example
-------
    from akida_client import AkidaServiceClient
    c = AkidaServiceClient("http://localhost:8790")   # compute-0
    c.stage_local_fbz("/path/to/your/model.fbz")      # only if not seeded
    c.load("vww_person_detect")
    print(c.health())
    print(c.infer([0] * (96 * 96 * 3)))               # VWW input is 96x96x3
    c.unload()
"""
from __future__ import annotations

import json
import os
import shutil
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))

DEFAULT_URL = os.environ.get("AKIDA_SERVICE_URL", "http://localhost:8790")
# Host side of the repo-local shared dir the compute containers bind-mount as
# /shared/models. Same env var and default as the dashboard, so the two agree.
DEFAULT_SHARED_MODELS = os.environ.get(
    "AKIDA_SHARED_MODELS", os.path.join(REPO, ".cluster", "shared", "models"))
# Sidecars the worker reads for class names; staged alongside the .fbz.
_SIDECAR_SUFFIXES = ("_meta.json", "_params.json", ".classes.json", ".json")


class AkidaServiceError(RuntimeError):
    pass


class AkidaServiceClient:
    def __init__(self, base_url: str = DEFAULT_URL,
                 shared_models_dir: str = DEFAULT_SHARED_MODELS,
                 timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.shared_models_dir = shared_models_dir
        self.timeout = timeout

    # ---- HTTP helpers ----
    def _get(self, path: str):
        req = urllib.request.Request(self.base_url + path)
        return self._send(req)

    def _post(self, path: str, body: dict | None = None):
        data = json.dumps(body or {}).encode("utf-8")
        req = urllib.request.Request(self.base_url + path, data=data,
                                     headers={"Content-Type": "application/json"})
        return self._send(req)

    def _send(self, req):
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            try:
                err = json.loads(e.read().decode("utf-8"))
            except Exception:
                err = {"error": "HTTP %s" % e.code}
            raise AkidaServiceError(err.get("error", str(err)))
        except Exception as e:
            raise AkidaServiceError("%s: %s" % (type(e).__name__, e))

    # ---- API ----
    def health(self) -> dict:
        return self._get("/health")

    def list_models(self) -> dict:
        return self._get("/models")

    def load(self, name: str) -> dict:
        return self._post("/load", {"name": name})

    def reload(self, name: str) -> dict:
        return self._post("/reload", {"name": name})

    def unload(self) -> dict:
        return self._post("/unload")

    def infer(self, values) -> dict:
        return self._post("/infer", {"input": list(values)})

    # ---- staging ----
    def stage_local_fbz(self, fbz_path: str) -> dict:
        """Copy a local .fbz (and any class-name sidecars) into the shared
        models dir so the worker can load it by name. Returns the staged
        names."""
        fbz_path = os.path.abspath(fbz_path)
        if not os.path.isfile(fbz_path) or not fbz_path.endswith(".fbz"):
            raise AkidaServiceError("not a .fbz file: %s" % fbz_path)
        os.makedirs(self.shared_models_dir, exist_ok=True)
        base = os.path.basename(fbz_path)[:-4]
        src_dir = os.path.dirname(fbz_path)
        staged = []
        dst = os.path.join(self.shared_models_dir, os.path.basename(fbz_path))
        shutil.copyfile(fbz_path, dst)
        staged.append(os.path.basename(dst))
        for suf in _SIDECAR_SUFFIXES:
            s = os.path.join(src_dir, base + suf)
            if os.path.isfile(s):
                shutil.copyfile(s, os.path.join(self.shared_models_dir, base + suf))
                staged.append(base + suf)
        return {"staged": staged, "models_dir": self.shared_models_dir}


if __name__ == "__main__":
    import sys
    c = AkidaServiceClient(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URL)
    print(json.dumps(c.health(), indent=2))
    print(json.dumps(c.list_models(), indent=2))
