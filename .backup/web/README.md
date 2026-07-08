# SymAkida control-plane GUI (runs on the laptop)

A Flask dashboard for the Akida model service running on the Symphony
compute nodes. It runs **natively on the laptop**, not on the cluster —
it talks to the per-node HTTP endpoints (`:8791/2/3`) to show fleet
status, list/load/unload/hot-swap models, stage a local `.fbz`, and run a
bundled sample dataset as a workload fanned across the live chips.

## Run

```bash
python3 -m venv .venv && ./.venv/bin/pip install flask
AKIDA_NODES="http://localhost:8791,http://localhost:8792,http://localhost:8793" \
FLASK_PORT=5001 ./.venv/bin/python app.py
# open http://localhost:5001
```

Env:

| Var | Default | Meaning |
|---|---|---|
| `AKIDA_NODES` | the three `localhost:879{1,2,3}` | comma-separated per-node service URLs |
| `AKIDA_SHARED_MODELS` | `/opt/symphony/shared/models` | host path the containers mount as `/shared/models` (used by "Stage local .fbz") |
| `FLASK_PORT` | `5001` | dashboard port |

The dashboard auto-detects which nodes are live, so it works whether the
cluster has one chip up or the full fleet. "Run across fleet" loads the
selected model on every live node and round-robins the sample inputs
across them, reporting per-sample node + latency and the class histogram.

Programmatic use without the GUI: `client/akida_client.py`
(`AkidaServiceClient`).
