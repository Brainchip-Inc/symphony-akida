"""Thin SOAM orchestrator for the image-shard-inference pipeline (Python 3.6).

Runs INSIDE the master container (needs Symphony's 3.6 soamapi binding). It does NO image math:
for each 448x448x3 input image it sends the bytes to the SegmentService, then drives the two
downstream Symphony stages, and reads back the final stitched detections. The image split, the
on-chip inference, and the box stitching all happen in the three SOAM services; Symphony
schedules and load-balances each stage across the cluster (the 5 segments of every image fan out
across the Akida chips).

The three stages run as a streaming pipeline (one SOAM session each, wired submit->fetch->submit),
bounded by an in-flight semaphore so /shared holds only a few images at a time:

  images --sem--> [Segment] --seg ack--> submit 5 --> [Inference x chips] --5 grids--> [Stitch] --> done

Inputs are real 448 samples prepared under /shared/samples (<model>.bin + sidecar) and sent as raw
bytes; a model with no sample set falls back to random uint8. Reports images/sec, per-chip
distribution and the fleet speedup over a single chip.

    run_client.sh --count 200
"""
from __future__ import print_function
import argparse
import array
import json
import os
import random
import shutil
import sys
import threading
import time
from collections import Counter, defaultdict

import soamapi

SEG_APP = os.environ.get("AKIDA_SEG_APP", "ShardSegmentService")
INF_APP = os.environ.get("AKIDA_INF_APP", "ShardInferenceService")
STITCH_APP = os.environ.get("AKIDA_STITCH_APP", "ShardStitchService")
MODELS_DIR = os.environ.get("AKIDA_MODELS_DIR", "/shared/models")
SAMPLES_DIR = os.environ.get("AKIDA_SAMPLES_DIR", "/shared/samples")
PIPE_DIR = os.environ.get("AKIDA_PIPELINE_DIR", "/shared/pipeline")
N_SEG = 5


class PipeMessage(soamapi.Message):
    """Shard pipeline wire format (MUST match the service containers):
       write_string(header_json); write_byte_array(array('B', payload), 0, len)."""

    def __init__(self, header=None, payload=b""):
        super(PipeMessage, self).__init__()
        self.header = header or {}
        self.payload = payload

    def on_serialize(self, stream):
        stream.write_string(json.dumps(self.header))
        arr = array.array("B", self.payload)
        stream.write_byte_array(arr, 0, len(arr))

    def on_deserialize(self, stream):
        self.header = json.loads(stream.read_string() or "{}")
        self.payload = stream.read_byte_array("B").tobytes()


def sample_length(model):
    """Bytes per input image = prod(sample_input_shape) (falls back to input_shape)."""
    meta = os.path.join(MODELS_DIR, model + "_meta.json")
    if not os.path.isfile(meta):
        raise SystemExit("no metadata for %s (expected %s)" % (model, meta))
    m = json.load(open(meta))
    shape = m.get("sample_input_shape") or m.get("input_shape")
    if not shape:
        raise SystemExit("no input_shape in %s" % meta)
    n = 1
    for d in shape:
        n *= int(d)
    return n, shape


def build_pool(model, n, count):
    """Raw-byte 448 images to cycle through: prefer prepared /shared/samples, else random."""
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
            pool = []
            with open(bin_p, "rb") as fh:
                for i in idx[:k]:
                    fh.seek(i * per)
                    pool.append(fh.read(per))
            return pool, "real (%d of %d)" % (k, avail)
        print("[client] %s sample set mismatch (per=%d n=%d); using random"
              % (model, per, n), file=sys.stderr)
    k = max(1, min(count, 64))
    pool = [bytes(bytearray(random.getrandbits(8) for _ in range(n))) for _ in range(k)]
    return pool, "random"


def _session(app, max_services=0):
    conn = soamapi.connect(app, soamapi.DefaultSecurityCallback("Admin", "Admin"))
    attrs = soamapi.SessionCreationAttributes()
    attrs.set_session_name("shard-" + app)
    attrs.set_session_type("UnrecoverableNoHistoricalData")
    attrs.set_session_flags(soamapi.SessionFlags.RECEIVE_SYNC)
    if max_services > 0:
        attrs.set_max_services(max_services)
    return conn, conn.create_session(attrs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo_akidanet_voc")
    ap.add_argument("--count", type=int, default=200, help="number of full 448 images")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--in-flight", type=int, default=0,
                    help="max images in the pipeline at once (0=auto from chip count)")
    ap.add_argument("--stall-timeout", type=float, default=60.0,
                    help="give up (report partial results) if no image completes for this long")
    ap.add_argument("--json", action="store_true", help="emit one JSON result line (dashboard)")
    args = ap.parse_args()

    nodes = int(os.environ.get("AKIDA_NUM_NODES", "0") or 0)
    in_flight = args.in_flight or (max(8, 3 * nodes) if nodes else 12)
    n, shape = sample_length(args.model)
    random.seed(args.seed)
    pool, source = build_pool(args.model, n, args.count)
    run_id = "r%d" % os.getpid()
    log_to = sys.stderr if args.json else sys.stdout
    print("[client] model=%s image_shape=%s images=%d in_flight=%d samples=%s"
          % (args.model, shape, args.count, in_flight, source), file=log_to)

    soamapi.initialize()
    seg_conn, seg_sess = _session(SEG_APP)
    inf_conn, inf_sess = _session(INF_APP, max_services=nodes)
    stitch_conn, stitch_sess = _session(STITCH_APP)

    sem = threading.BoundedSemaphore(in_flight)
    state_lock = threading.Lock()
    img_state = {}            # image_id -> {"got": int, "err": bool, "done": bool}
    per_host = Counter()
    per_host_us = defaultdict(float)
    classes = Counter()
    totals = {"seg_err": 0, "inf_done": 0, "inf_err": 0, "img_done": 0, "img_err": 0, "boxes": 0}
    progress = {"last": time.time()}
    stalled = {"v": False}
    done_evt = threading.Event()

    def _finish(image_id, failed):
        # Guard against double-finish (a duplicate/retried task output must not release the
        # semaphore twice, which would raise on the BoundedSemaphore and strand the pipeline).
        with state_lock:
            st = img_state.get(image_id)
            if st is not None and st.get("done"):
                return
            if st is not None:
                st["done"] = True
            totals["img_done"] += 1
            if failed:
                totals["img_err"] += 1
            progress["last"] = time.time()
            if totals["img_done"] >= args.count:
                done_evt.set()
        shutil.rmtree(os.path.join(PIPE_DIR, image_id), ignore_errors=True)
        sem.release()

    # stage 1: submit one segment task per image (bounded by the in-flight semaphore)
    def seg_submit():
        for i in range(args.count):
            sem.acquire()
            image_id = "%s_%d" % (run_id, i)
            with state_lock:
                img_state[image_id] = {"got": 0, "err": False, "done": False, "dispatched": False}
            tsa = soamapi.TaskSubmissionAttributes()
            tsa.set_task_input(PipeMessage({"image_id": image_id, "model": args.model},
                                           pool[i % len(pool)]))
            seg_sess.send_task_input(tsa)

    # stage 1 fetch -> stage 2 submit (5 inference tasks per segmented image)
    def seg_fetch():
        seen = 0
        # fetch ONE output at a time: in a streaming pipeline only `in_flight` tasks are ever
        # outstanding, so requesting the full count would block waiting for outputs that are
        # gated behind downstream completion -> deadlock. One-at-a-time returns as soon as any
        # segment is ready, so inference can be submitted immediately.
        while seen < args.count:
            for out in seg_sess.fetch_task_output(1, 30):
                seen += 1
                r = _reply(out)
                image_id = r.get("image_id")
                if not out.is_successful() or not r.get("ok"):
                    with state_lock:
                        totals["seg_err"] += 1
                    _finish(image_id, True)
                    continue
                for k in range(N_SEG):
                    tsa = soamapi.TaskSubmissionAttributes()
                    tsa.set_task_input(PipeMessage(
                        {"image_id": image_id, "seg_idx": k, "model": args.model}))
                    inf_sess.send_task_input(tsa)

    # stage 2 fetch -> stage 3 submit (one stitch task once all 5 grids are in)
    def inf_fetch():
        seen = 0
        target = args.count * N_SEG
        while seen < target:
            for out in inf_sess.fetch_task_output(1, 30):   # one at a time (see seg_fetch)
                seen += 1
                r = _reply(out)
                image_id = r.get("image_id")
                ok = out.is_successful() and r.get("ok")
                with state_lock:
                    totals["inf_done"] += 1
                    if ok:
                        per_host[r["host"]] += 1
                        per_host_us[r["host"]] += r.get("inference_us", 0)
                    else:
                        totals["inf_err"] += 1
                    st = img_state.get(image_id)
                    if st is None:
                        continue
                    st["got"] += 1
                    if not ok:
                        st["err"] = True
                    # dispatch the next stage exactly once, when all 5 grids are in (a
                    # duplicate/retried inference output must not re-trigger it)
                    act = None
                    if st["got"] >= N_SEG and not st["dispatched"]:
                        st["dispatched"] = True
                        act = "fail" if st["err"] else "stitch"
                if act == "fail":
                    _finish(image_id, True)
                elif act == "stitch":
                    tsa = soamapi.TaskSubmissionAttributes()
                    tsa.set_task_input(PipeMessage({"image_id": image_id, "model": args.model}))
                    stitch_sess.send_task_input(tsa)

    # stage 3 fetch -> record final detections, clean up, release the in-flight slot
    def stitch_fetch():
        seen = 0
        # only images whose 5 inferences all succeeded reach this stage
        while not done_evt.is_set():
            for out in stitch_sess.fetch_task_output(1, 5):   # one at a time (see seg_fetch)
                seen += 1
                r = _reply(out)
                ok = out.is_successful() and r.get("ok")
                if ok:
                    with state_lock:
                        totals["boxes"] += int(r.get("n_boxes", 0))
                        for name, c in (r.get("class_hist") or {}).items():
                            classes[name] += c
                _finish(r.get("image_id"), not ok)

    # watchdog: if no image completes for --stall-timeout seconds the fleet has likely stalled
    # (e.g. a chip stuck under sustained load); abandon the in-flight images and report partial
    # results instead of hanging forever.
    def watchdog():
        while not done_evt.is_set():
            time.sleep(3)
            with state_lock:
                idle = time.time() - progress["last"]
                pending = args.count - totals["img_done"]
            if pending > 0 and idle > args.stall_timeout:
                stalled["v"] = True
                done_evt.set()
                return

    t0 = time.time()
    with state_lock:
        progress["last"] = t0
    threads = [threading.Thread(target=f)
               for f in (seg_submit, seg_fetch, inf_fetch, stitch_fetch, watchdog)]
    for th in threads:
        th.daemon = True
        th.start()
    done_evt.wait()
    wall = time.time() - t0

    # all images finished -> the seg/inf fetch loops have already hit their targets; give the
    # stitch loop a moment to fall out of its poll before tearing the sessions down.
    for th in threads:
        th.join(timeout=8)

    for c, s in ((seg_conn, seg_sess), (inf_conn, inf_sess), (stitch_conn, stitch_sess)):
        try:
            s.close(); c.close()
        except Exception:
            pass
    soamapi.uninitialize()

    ok_img = totals["img_done"] - totals["img_err"]
    seg_ok = sum(per_host.values())
    avg_seg_ms = (sum(per_host_us.values()) / seg_ok / 1000.0) if seg_ok else 0.0
    rate = (args.count / wall) if wall else 0.0
    # a single chip runs the 5 segments serially -> its images/sec = 1 / (5 * avg segment latency)
    one_chip = (1000.0 / (N_SEG * avg_seg_ms)) if avg_seg_ms else 0.0
    result = {
        "model": args.model, "count": args.count, "input_source": source,
        "stalled": stalled["v"],
        "images_done": ok_img, "image_errors": totals["img_err"],
        "segments_done": seg_ok, "segment_errors": totals["inf_err"] + totals["seg_err"],
        "chips": len(per_host), "wall_s": round(wall, 3),
        "throughput": round(rate, 2), "avg_seg_ms": round(avg_seg_ms, 3),
        "one_chip_rate": round(one_chip, 2),
        "speedup": round(rate / one_chip, 2) if one_chip else 0.0,
        "avg_boxes": round(totals["boxes"] / ok_img, 1) if ok_img else 0.0,
        "per_host": {h: {"tasks": per_host[h],
                         "avg_ms": round(per_host_us[h] / per_host[h] / 1000.0, 3)}
                     for h in per_host},
        "classes": dict(classes),
    }
    if args.json:
        print(json.dumps(result))
        return

    print("\n=== shard pipeline across the Akida fleet ===")
    if stalled["v"]:
        print("!! STALLED: no image completed for %.0fs (fleet likely stuck under load); "
              "partial results below. Tear down + relaunch (reset chips) and retry a smaller batch."
              % args.stall_timeout)
    print("input images:  %s (%dx%dx%d, sharded into %d)" % (source, shape[0], shape[1], shape[2], N_SEG))
    print("chips used:    %d" % result["chips"])
    print("images:        %d done, %d error" % (result["images_done"], result["image_errors"]))
    print("segments:      %d done, %d error" % (result["segments_done"], result["segment_errors"]))
    print("wall time:     %.2f s" % wall)
    print("throughput:    %.2f images/sec  (avg %.1f boxes/image)" % (rate, result["avg_boxes"]))
    print("\nper-chip segment distribution:")
    for h in sorted(per_host):
        print("  %-26s %6d segments   avg on-chip %.2f ms"
              % (h, per_host[h], result["per_host"][h]["avg_ms"]))
    print("\navg on-chip latency/segment: %.2f ms" % avg_seg_ms)
    print("one chip (5 segments serial): ~%.2f images/sec" % one_chip)
    print("fleet of %d chips:            %.2f images/sec  (~%.1fx a single chip)"
          % (len(per_host), rate, result["speedup"]))
    print("\nclass histogram: %s" % dict(classes))


def _reply(out):
    if not out.is_successful():
        return {}
    msg = soamapi.DefaultTextMessage()
    out.populate_task_output(msg)
    try:
        return json.loads(msg.get_text())
    except Exception:
        return {}


if __name__ == "__main__":
    main()
