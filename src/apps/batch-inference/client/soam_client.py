"""SOAM batch client: fan a batch of inferences across the Akida fleet.

Runs INSIDE the master container (it needs Symphony's Python 3.6 soamapi binding
and the cluster security context). It opens a session against AkidaGenericService
and submits one task per input sample; Symphony's session manager fans the tasks
out across every compute node's Akida chip in parallel. It then reports the
per-chip task distribution and throughput -- the multi-Akida advantage, measured.

    run_client.sh --model kws_keyword_spotting --count 500
"""
from __future__ import print_function
import argparse
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict

import soamapi

APP = os.environ.get("AKIDA_APP", "AkidaGenericService")
MODELS_DIR = os.environ.get("AKIDA_MODELS_DIR", "/shared/models")


def input_length(model):
    meta = os.path.join(MODELS_DIR, model + "_meta.json")
    if not os.path.isfile(meta):
        raise SystemExit("no metadata for %s (expected %s)" % (model, meta))
    shape = json.load(open(meta)).get("input_shape")
    if not shape:
        raise SystemExit("no input_shape in %s" % meta)
    n = 1
    for d in shape:
        n *= int(d)
    return n, shape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="kws_keyword_spotting")
    ap.add_argument("--count", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--json", action="store_true", help="emit one JSON result line (for the dashboard)")
    args = ap.parse_args()

    n, shape = input_length(args.model)
    random.seed(args.seed)
    samples = [[random.randint(0, 255) for _ in range(n)] for _ in range(args.count)]
    print("[client] model=%s input_shape=%s tasks=%d -> %s"
          % (args.model, shape, args.count, APP),
          file=(sys.stderr if args.json else sys.stdout))

    soamapi.initialize()
    conn = soamapi.connect(APP, soamapi.DefaultSecurityCallback("Admin", "Admin"))
    attrs = soamapi.SessionCreationAttributes()
    attrs.set_session_name("akida-batch")
    attrs.set_session_type("UnrecoverableNoHistoricalData")
    attrs.set_session_flags(soamapi.SessionFlags.RECEIVE_SYNC)
    session = conn.create_session(attrs)

    t0 = time.time()
    for s in samples:
        tsa = soamapi.TaskSubmissionAttributes()
        msg = soamapi.DefaultTextMessage()
        msg.set_text(json.dumps({"model": args.model, "input": s}))
        tsa.set_task_input(msg)
        session.send_task_input(tsa)

    per_host = Counter()
    per_host_us = defaultdict(float)
    classes = Counter()
    errors = 0
    done = 0
    while done < args.count:
        for out in session.fetch_task_output(args.count, 120):
            done += 1
            if not out.is_successful():
                errors += 1
                continue
            reply = soamapi.DefaultTextMessage()
            out.populate_task_output(reply)
            r = json.loads(reply.get_text())
            if "error" in r:
                errors += 1
                continue
            per_host[r["host"]] += 1
            per_host_us[r["host"]] += r.get("inference_us", 0)
            classes[r.get("cls_name", "?")] += 1
    wall = time.time() - t0

    session.close()
    conn.close()
    soamapi.uninitialize()

    rate = (args.count / wall) if wall else 0.0
    ok = sum(per_host.values())
    avg_ms = (sum(per_host_us.values()) / ok / 1000.0) if ok else 0.0
    one_chip = (1000.0 / avg_ms) if avg_ms else 0.0
    result = {
        "model": args.model, "count": args.count,
        "done": args.count - errors, "errors": errors,
        "chips": len(per_host), "wall_s": round(wall, 3),
        "throughput": round(rate, 1), "avg_ms": round(avg_ms, 3),
        "one_chip_rate": round(one_chip, 1),
        "speedup": round(rate / one_chip, 2) if one_chip else 0.0,
        "per_host": {h: {"tasks": per_host[h],
                         "avg_ms": round(per_host_us[h] / per_host[h] / 1000.0, 3)}
                     for h in per_host},
        "classes": dict(classes),
    }
    if args.json:
        print(json.dumps(result))
        return

    print("\n=== fan-out across the Akida fleet ===")
    print("chips used:   %d" % result["chips"])
    print("tasks:        %d done, %d error" % (result["done"], errors))
    print("wall time:    %.2f s" % wall)
    print("throughput:   %.1f inferences/sec" % rate)
    print("\nper-chip distribution:")
    for h in sorted(per_host):
        print("  %-26s %6d tasks   avg on-chip %.2f ms"
              % (h, per_host[h], result["per_host"][h]["avg_ms"]))
    print("\navg on-chip latency: %.2f ms  ->  one chip sustains ~%.0f inf/s" % (avg_ms, one_chip))
    print("fleet of %d chips:    %.0f inf/s  (~%.1fx a single chip)"
          % (len(per_host), rate, result["speedup"]))
    print("\nclass histogram: %s" % dict(classes))


if __name__ == "__main__":
    main()
