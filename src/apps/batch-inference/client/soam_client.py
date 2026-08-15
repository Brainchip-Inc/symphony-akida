"""SOAM batch client: fan a batch of inferences across the Akida fleet.

Runs INSIDE the master container (it needs Symphony's Python 3.6 soamapi binding
and the cluster security context). It opens a session against AkidaGenericService
and submits one task per input sample; Symphony's session manager fans the tasks
out across every compute node's Akida chip in parallel. It then reports the
per-chip task distribution and throughput -- the multi-Akida advantage, measured.

Inputs are real samples prepared under /shared/samples (<model>.bin + sidecar) and
sent as raw bytes (binary soamapi message), not JSON int arrays -- ~4x smaller on
the wire and no parsing. A model with no sample set falls back to random uint8.

    run_client.sh --model kws_keyword_spotting_sparse --count 500
"""
from __future__ import print_function
import argparse
import array
import json
import os
import random
import sys
import threading
import time
from collections import Counter, defaultdict

import soamapi

APP = os.environ.get("AKIDA_APP", "AkidaGenericService")
MODELS_DIR = os.environ.get("AKIDA_MODELS_DIR", "/shared/models")
SAMPLES_DIR = os.environ.get("AKIDA_SAMPLES_DIR", "/shared/samples")


class TensorInputMessage(soamapi.Message):
    """Binary task input. Wire format MUST match AkidaServiceContainer.py:
       write_string(model); write_byte_array(array('B', tensor_bytes), 0, len)."""

    def __init__(self, model="", data=b""):
        super(TensorInputMessage, self).__init__()
        self.model = model
        self.data = data  # raw uint8 tensor bytes

    def on_serialize(self, stream):
        arr = array.array("B", self.data)
        stream.write_string(self.model or "")
        stream.write_byte_array(arr, 0, len(arr))

    def on_deserialize(self, stream):
        self.model = stream.read_string()
        self.data = stream.read_byte_array("B").tobytes()


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


def build_pool(model, n, count):
    """A pool of raw-byte tensors to cycle through. Prefer real samples from
    /shared/samples (<model>.bin + sidecar, read with the stdlib -- the 3.6
    client has no numpy); otherwise fall back to random uint8 of the right size."""
    base = os.path.join(SAMPLES_DIR, model)
    side_p, bin_p = base + ".samples.json", base + ".bin"
    if os.path.isfile(side_p) and os.path.isfile(bin_p):
        side = json.load(open(side_p))
        per = int(side.get("per_sample_bytes", 0))
        avail = int(side.get("count", 0))
        if per == n and avail > 0:
            k = max(1, min(avail, count))
            idx = list(range(avail))
            random.shuffle(idx)
            idx = idx[:k]
            pool = []
            with open(bin_p, "rb") as fh:
                for i in idx:
                    fh.seek(i * per)
                    pool.append(fh.read(per))
            return pool, "real (%d of %d)" % (k, avail)
        print("[client] %s sample set mismatches model (per=%d n=%d); using random"
              % (model, per, n), file=sys.stderr)
    k = max(1, min(count, 256))
    pool = [bytes(bytearray(random.getrandbits(8) for _ in range(n))) for _ in range(k)]
    return pool, "random"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="kws_keyword_spotting_sparse")
    ap.add_argument("--count", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max-services", type=int, default=0,
                    help="cap service instances used (0=unlimited); set to the chip count so "
                         "one session uses one instance per chip and does not over-provision")
    ap.add_argument("--workers", type=int, default=8,
                    help="concurrent submit threads; a single thread cannot feed a fast fleet, "
                         "so several keep a backlog and every chip stays saturated")
    ap.add_argument("--json", action="store_true", help="emit one JSON result line (for the dashboard)")
    args = ap.parse_args()

    n, shape = input_length(args.model)
    random.seed(args.seed)
    # Pre-build a pool of byte tensors once, then cycle through them. Cheap per-task
    # submit + several sender threads let the client build a backlog and keep every
    # chip saturated (a single thread cannot feed a fast fleet).
    pool, source = build_pool(args.model, n, args.count)
    print("[client] model=%s input_shape=%s tasks=%d samples=%s -> %s"
          % (args.model, shape, args.count, source, APP),
          file=(sys.stderr if args.json else sys.stdout))

    soamapi.initialize()
    conn = soamapi.connect(APP, soamapi.DefaultSecurityCallback("Admin", "Admin"))
    attrs = soamapi.SessionCreationAttributes()
    attrs.set_session_name("akida-batch")
    attrs.set_session_type("UnrecoverableNoHistoricalData")
    attrs.set_session_flags(soamapi.SessionFlags.RECEIVE_SYNC)
    if args.max_services > 0:
        attrs.set_max_services(args.max_services)
    session = conn.create_session(attrs)

    def submit(lo, hi):
        for i in range(lo, hi):
            tsa = soamapi.TaskSubmissionAttributes()
            tsa.set_task_input(TensorInputMessage(args.model, pool[i % len(pool)]))
            session.send_task_input(tsa)

    t0 = time.time()
    workers = max(1, args.workers)
    step = (args.count + workers - 1) // workers
    senders = []
    for w in range(workers):
        lo, hi = w * step, min(args.count, (w + 1) * step)
        if lo >= hi:
            break
        th = threading.Thread(target=submit, args=(lo, hi))
        th.start()
        senders.append(th)

    per_host = Counter()
    per_host_us = defaultdict(float)
    # Which chip each host turned out to be. Every reply already carries it (akida_chip's
    # Chip.infer identity dict); it used to be parsed and dropped, so the dashboard could
    # only ever name the hosts, never the silicon. First reply per host wins -- a node is
    # pinned to one chip for the life of the container, so there is nothing to update.
    per_host_id = {}
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
            per_host_id.setdefault(r["host"], {"device": r.get("device"),
                                               "product": r.get("product"),
                                               "model": r.get("model")})
            classes[r.get("cls_name", "?")] += 1
    wall = time.time() - t0
    for th in senders:
        th.join()

    session.close()
    conn.close()
    soamapi.uninitialize()

    rate = (args.count / wall) if wall else 0.0
    ok = sum(per_host.values())
    avg_ms = (sum(per_host_us.values()) / ok / 1000.0) if ok else 0.0
    one_chip = (1000.0 / avg_ms) if avg_ms else 0.0
    result = {
        "model": args.model, "count": args.count, "input_source": source,
        "done": args.count - errors, "errors": errors,
        "chips": len(per_host), "wall_s": round(wall, 3),
        "throughput": round(rate, 1), "avg_ms": round(avg_ms, 3),
        "one_chip_rate": round(one_chip, 1),
        "speedup": round(rate / one_chip, 2) if one_chip else 0.0,
        "per_host": {h: dict(per_host_id.get(h, {}),
                             tasks=per_host[h],
                             avg_ms=round(per_host_us[h] / per_host[h] / 1000.0, 3))
                     for h in per_host},
        "classes": dict(classes),
    }
    if args.json:
        print(json.dumps(result))
        return

    print("\n=== fan-out across the Akida fleet ===")
    print("input samples: %s" % source)
    print("chips used:   %d" % result["chips"])
    print("tasks:        %d done, %d error" % (result["done"], errors))
    print("wall time:    %.2f s" % wall)
    print("throughput:   %.1f inferences/sec" % rate)
    print("\nper-chip distribution:")
    for h in sorted(per_host):
        entry = result["per_host"][h]
        print("  %-26s %-9s %6d tasks   avg on-chip %.2f ms"
              % (h, entry.get("product") or "?", per_host[h], entry["avg_ms"]))
    print("\navg on-chip latency: %.2f ms  ->  one chip sustains ~%.0f inf/s" % (avg_ms, one_chip))
    print("fleet of %d chips:    %.0f inf/s  (~%.1fx a single chip)"
          % (len(per_host), rate, result["speedup"]))
    print("\nclass histogram: %s" % dict(classes))


if __name__ == "__main__":
    main()
