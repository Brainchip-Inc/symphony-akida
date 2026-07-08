#!/usr/bin/env python3
"""Command-line demo: load an Akida .fbz on the Symphony fleet and test it.

Uses AkidaServiceClient (no akida SDK needed on the laptop — the model
runs on the cluster). Lists models, loads one, runs inference, and (if a
bundled sample dataset exists) replays it and prints a class histogram.

Examples
--------
    # list what's available on the service
    python client/demo.py --list

    # load voice_auth and run its bundled sample dataset
    python client/demo.py --model voice_auth

    # load a model and run one explicit input vector
    python client/demo.py --model esm_classifier --infer "$(python3 -c 'import json;print(json.dumps([7]*256))')"

    # point at a different node / stage a local .fbz first
    python client/demo.py --url http://localhost:8792 --stage /path/to/mymodel.fbz --model mymodel
"""
import argparse
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from akida_client import AkidaServiceClient, AkidaServiceError, DEFAULT_URL  # noqa: E402

SAMPLES_DIR = os.path.join(HERE, "..", "samples")


def _samples_for(model_name):
    """Find a bundled sample dataset for a model (by base name)."""
    base = model_name[:-4] if model_name.endswith(".fbz") else model_name
    path = os.path.join(SAMPLES_DIR, base + ".samples.json")
    if os.path.isfile(path):
        with open(path) as fh:
            return json.load(fh)["samples"]
    return None


def main():
    ap = argparse.ArgumentParser(description="Load + test an Akida model on the Symphony fleet")
    ap.add_argument("--url", default=DEFAULT_URL, help="service node URL (default %(default)s)")
    ap.add_argument("--list", action="store_true", help="list available models and exit")
    ap.add_argument("--stage", metavar="FBZ", help="stage a local .fbz into the cluster first")
    ap.add_argument("--model", help="model name to load (e.g. voice_auth)")
    ap.add_argument("--infer", metavar="JSON", help="one input vector as a JSON list")
    ap.add_argument("--unload", action="store_true", help="unload after testing")
    args = ap.parse_args()

    c = AkidaServiceClient(args.url)
    try:
        h = c.health()
    except AkidaServiceError as e:
        print("cannot reach service at %s: %s" % (args.url, e)); return 2
    print("service: %s  akida %s  models_dir=%s" % (h["host"], h["akida_version"], h["models_dir"]))

    if args.stage:
        print("staging %s ..." % args.stage)
        print("  staged:", ", ".join(c.stage_local_fbz(args.stage)["staged"]))

    if args.list or not args.model:
        models = c.list_models()["models"]
        print("\navailable models:")
        for m in models:
            print("  %-32s %-10s %s" % (m["name"], "x".join(str(d) for d in m.get("input_shape", [])),
                                        ",".join(m.get("class_names", [])[:6])))
        if not args.model:
            return 0

    # load
    meta = c.load(args.model)["model"]
    print("\nloaded %s  input=%s  classes=%s  akida_mapped=%s"
          % (meta["name"], meta["input_shape"], meta["class_names"], meta["akida_mapped"]))

    # explicit single inference
    if args.infer:
        r = c.infer(json.loads(args.infer))
        print("infer -> %s  (cls=%d, %d us, on %s)" % (r["cls_name"], r["cls"], r["inference_us"], r["host"]))

    # bundled sample dataset replay
    samples = _samples_for(meta["name"])
    if samples:
        print("\nreplaying %d bundled samples:" % len(samples))
        hist, lat = collections.Counter(), []
        for s in samples:
            r = c.infer(s)
            hist[r["cls_name"]] += 1
            lat.append(r["inference_us"])
        for k, v in hist.most_common():
            print("  %-14s %s (%d)" % (k, "#" * v, v))
        print("  avg %d us/inference over %d samples on %s" % (sum(lat) // len(lat), len(lat), r["host"]))
    elif not args.infer:
        print("(no bundled samples for %s; pass --infer to test one input)" % meta["name"])

    if args.unload:
        c.unload(); print("\nunloaded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
