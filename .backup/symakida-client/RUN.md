# SymAkida laptop client + dashboard — standalone bundle

Talks to a running `symphonyce:7.3.4` Akida cluster over HTTP. No repo,
no akida SDK, and no `requests` needed on this side — `akida_client.py`
is pure Python stdlib. The only dependency is Flask (for the dashboard).

## Layout (do not flatten — app.py imports ../client and reads ../samples)

    symakida-client/
    ├── web/app.py             dashboard (Flask, port 5001; HTML is inline)
    ├── client/akida_client.py HTTP client lib  (mandatory)
    ├── client/demo.py         CLI: list / load / replay samples
    ├── samples/*.samples.json bundled workloads for "Run across fleet"
    └── requirements.txt       flask

## Setup

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

## Run the dashboard

    cd web && python app.py            # http://localhost:5001
    # point it at a compute node's HTTP endpoint in the UI, e.g.
    # http://<cluster-host>:8791

`AKIDA_SHARED_MODELS` (default `/opt/symphony/shared/models`) only
matters for the "stage a local .fbz" button; set it if the shared
models dir is elsewhere.

## Run the CLI instead

    cd client
    # default node is http://localhost:8791 (override with --url or AKIDA_SERVICE_URL)
    python demo.py --url http://<cluster-host>:8791 --list
    python demo.py --url http://<cluster-host>:8791 --model voice_auth          # load + test (replays bundled samples)
    python demo.py --url http://<cluster-host>:8791 --stage /path/to/model.fbz --model mymodel
    python demo.py --url http://<cluster-host>:8791 --model voice_auth --infer '[0,1,2,...]'
    python demo.py --url http://<cluster-host>:8791 --model voice_auth --unload

Flags: `--url --list --stage FBZ --model NAME --infer JSON --unload`
(run `python demo.py -h` to confirm).
