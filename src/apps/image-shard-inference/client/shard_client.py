"""Thin SOAM orchestrator for the image-shard-inference pipeline (Python 3.6).

Runs INSIDE the master container (needs Symphony's 3.6 soamapi binding). It does NO image
math: for each 448x448x3 input frame it sends the bytes to the SegmentService, then drives the
two downstream Symphony stages, and reads back the final merged detections. The tile split,
the on-chip inference and the detection merge all happen in the three SOAM services; Symphony
schedules and load-balances each stage across the cluster, so the six tiles of every frame fan
out across the Akida chips.

The three stages run as a streaming pipeline (one SOAM session each, wired submit->fetch->
submit), bounded by an in-flight semaphore so /shared holds only a few frames at a time:

  frames --sem--> [Segment] --ack--> submit 6 --> [Inference x chips] --acks--> [Stitch] --> done

Frames come from a prepared sample set under /shared/samples (<set>.bin + sidecar) and are
sent as raw bytes; with no sample set it falls back to random uint8, which exercises every
stage and every throughput number but contains no objects to find. Reports frames/sec,
per-chip distribution and the fleet speedup over a single chip, and can dump the merged
detections for scoring.

    run_client.sh --count 100
    run_client.sh --samples voc2007 --count 100 --ordered --post-thresh 0 --dump
"""
from __future__ import print_function
import argparse
import json
import os
import random
import shutil
import sys
import threading
import time
from collections import Counter, defaultdict

import soamapi

from shard_wire import PipeMessage

SEG_APP = os.environ.get("AKIDA_SEG_APP", "ShardSegmentService")
INF_APP = os.environ.get("AKIDA_INF_APP", "ShardInferenceService")
STITCH_APP = os.environ.get("AKIDA_STITCH_APP", "ShardStitchService")
MODELS_DIR = os.environ.get("AKIDA_MODELS_DIR", "/shared/models")
SAMPLES_DIR = os.environ.get("AKIDA_SAMPLES_DIR", "/shared/samples")
PIPE_DIR = os.environ.get("AKIDA_PIPELINE_DIR", "/shared/pipeline")
RESULTS_DIR = os.environ.get("AKIDA_RESULTS_DIR", "/shared/results")


def load_meta(model):
    path = os.path.join(MODELS_DIR, model + "_meta.json")
    if not os.path.isfile(path):
        raise SystemExit("no metadata for %s (expected %s)" % (model, path))
    return json.load(open(path))


def frame_bytes(meta):
    """Bytes per input frame = prod(sample_input_shape) (falls back to input_shape)."""
    shape = meta.get("sample_input_shape") or meta.get("input_shape")
    total = 1
    for dim in shape:
        total *= int(dim)
    return total, shape


class FramePool(object):
    """Frames to send, read from the .bin on demand.

    Deliberately not preloaded: the full VOC2007 test split is 4,952 x 602 KB, so holding it
    would cost ~3 GB of the master's RSS to save a seek per frame.
    """

    def __init__(self, name, per_frame, count, ordered, seed):
        base = os.path.join(SAMPLES_DIR, name)
        self.path = base + ".bin"
        self.random = None
        sidecar = base + ".samples.json"
        if os.path.isfile(sidecar) and os.path.isfile(self.path):
            side = json.load(open(sidecar))
            if int(side.get("per_sample_bytes", 0)) == per_frame and side.get("count"):
                self.available = int(side["count"])
                self.source = side.get("source_npz") or name
                self.indices = list(range(self.available))
                if not ordered:
                    random.Random(seed).shuffle(self.indices)
                self.indices = self.indices[:count] or [0]
                self.per_frame = per_frame
                self.handle = open(self.path, "rb")
                self.description = "%s (%d of %d%s)" % (name, len(self.indices),
                                                        self.available,
                                                        "" if ordered else ", shuffled")
                return
            print("[client] %s sample set mismatch (per=%s want=%d); using random"
                  % (name, side.get("per_sample_bytes"), per_frame), file=sys.stderr)
        self.available = 0
        self.source = None
        self.per_frame = per_frame
        self.handle = None
        pool_size = max(1, min(count, 64))
        rng = random.Random(seed)
        self.random = [bytes(bytearray(rng.getrandbits(8) for _ in range(per_frame)))
                       for _ in range(pool_size)]
        self.indices = [i % pool_size for i in range(count)]
        self.description = "random uint8 (no objects to detect)"

    @property
    def is_random(self):
        return self.random is not None

    def read(self, i):
        index = self.indices[i % len(self.indices)]
        if self.random is not None:
            return index, self.random[index]
        self.handle.seek(index * self.per_frame)
        return index, self.handle.read(self.per_frame)


def _session(app, max_services=0):
    conn = soamapi.connect(app, soamapi.DefaultSecurityCallback("Admin", "Admin"))
    attrs = soamapi.SessionCreationAttributes()
    attrs.set_session_name("shard-" + app)
    attrs.set_session_type("UnrecoverableNoHistoricalData")
    attrs.set_session_flags(soamapi.SessionFlags.RECEIVE_SYNC)
    if max_services > 0:
        attrs.set_max_services(max_services)
    return conn, conn.create_session(attrs)


def _reply(out):
    if not out.is_successful():
        return {}
    msg = soamapi.DefaultTextMessage()
    out.populate_task_output(msg)
    try:
        return json.loads(msg.get_text())
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tiled_yolov2_voc")
    ap.add_argument("--samples", default=None,
                    help="prepared sample set name (default: the model's own set)")
    ap.add_argument("--count", type=int, default=200, help="number of full 448 frames")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--ordered", action="store_true",
                    help="send frames in dataset order, so frame i is sample i (needed to "
                         "score a dump against ground truth)")
    ap.add_argument("--post-thresh", type=float, default=None,
                    help="post-merge score gate (default: the model's post_merge_thresh; "
                         "pass 0 to reproduce the reference mAP protocol)")
    ap.add_argument("--max-boxes", type=int, default=None,
                    help="cap on merged detections per frame (default: the model's max_boxes)")
    ap.add_argument("--dump", nargs="?", const="", default=None,
                    help="write merged detections as JSONL for scoring (default path under "
                         "/shared/results)")
    ap.add_argument("--in-flight", type=int, default=0,
                    help="max frames in the pipeline at once (0=auto from chip count)")
    ap.add_argument("--stall-timeout", type=float, default=60.0,
                    help="give up (report partial results) if no frame completes for this long")
    ap.add_argument("--json", action="store_true", help="emit one JSON result line (dashboard)")
    args = ap.parse_args()

    meta = load_meta(args.model)
    n_tiles = _tile_count(meta)
    per_frame, shape = frame_bytes(meta)
    nodes = int(os.environ.get("AKIDA_NUM_NODES", "0") or 0)
    in_flight = args.in_flight or (max(8, 3 * nodes) if nodes else 12)
    pool = FramePool(args.samples or args.model, per_frame, args.count, args.ordered,
                     args.seed)
    run_id = "r%d" % os.getpid()
    log_to = sys.stderr if args.json else sys.stdout
    print("[client] model=%s frame=%s frames=%d tiles=%d in_flight=%d samples=%s"
          % (args.model, shape, args.count, n_tiles, in_flight, pool.description),
          file=log_to)

    dump_path = None
    if args.dump is not None:
        # Named after the sample SET, not the model: the scorer pairs a dump with the test kit
        # it came from, and several sets feed the same model.
        dump_path = args.dump or os.path.join(
            RESULTS_DIR, "%s_%s.jsonl" % (args.samples or args.model, run_id))
        try:
            os.makedirs(os.path.dirname(dump_path))
        except OSError:
            pass
        print("[client] dumping detections to %s" % dump_path, file=log_to)

    stitch_header = {"model": args.model}
    if args.post_thresh is not None:
        stitch_header["post_thresh"] = args.post_thresh
    if args.max_boxes is not None:
        stitch_header["max_boxes"] = args.max_boxes

    soamapi.initialize()
    seg_conn, seg_sess = _session(SEG_APP)
    inf_conn, inf_sess = _session(INF_APP, max_services=nodes)
    stitch_conn, stitch_sess = _session(STITCH_APP)

    sem = threading.BoundedSemaphore(in_flight)
    state_lock = threading.Lock()
    dump_lock = threading.Lock()
    dump_file = open(dump_path, "w") if dump_path else None
    img_state = {}            # image_id -> {"got", "err", "done", "dispatched", "sample"}
    per_host = Counter()
    per_host_us = defaultdict(float)
    # Which chip each host turned out to be. Every tile reply already carries it (akida_chip's
    # Chip.predict_tile identity dict); it used to be parsed and dropped, so the dashboard could
    # only ever name the hosts, never the silicon. First reply per host wins -- a node is
    # pinned to one chip for the life of the container, so there is nothing to update.
    per_host_id = {}
    classes = Counter()
    totals = {"seg_err": 0, "inf_done": 0, "inf_err": 0, "img_done": 0, "img_err": 0,
              "boxes": 0, "decode_us": 0.0}
    progress = {"last": time.time()}
    stalled = {"v": False}
    done_evt = threading.Event()

    def _finish(image_id, failed):
        # Guard against double-finish (a duplicate/retried task output must not release the
        # semaphore twice, which would raise on the BoundedSemaphore and strand the pipeline).
        with state_lock:
            state = img_state.get(image_id)
            if state is not None and state.get("done"):
                return
            if state is not None:
                state["done"] = True
            totals["img_done"] += 1
            if failed:
                totals["img_err"] += 1
            progress["last"] = time.time()
            if totals["img_done"] >= args.count:
                done_evt.set()
        shutil.rmtree(os.path.join(PIPE_DIR, image_id), ignore_errors=True)
        sem.release()

    # stage 1: submit one segment task per frame (bounded by the in-flight semaphore)
    def seg_submit():
        for i in range(args.count):
            sem.acquire()
            image_id = "%s_%d" % (run_id, i)
            sample, payload = pool.read(i)
            with state_lock:
                img_state[image_id] = {"got": 0, "err": False, "done": False,
                                       "dispatched": False, "sample": sample}
            tsa = soamapi.TaskSubmissionAttributes()
            tsa.set_task_input(PipeMessage({"image_id": image_id, "model": args.model},
                                           payload))
            seg_sess.send_task_input(tsa)

    # stage 1 fetch -> stage 2 submit (one inference task per tile of each split frame)
    def seg_fetch():
        seen = 0
        # fetch ONE output at a time: in a streaming pipeline only `in_flight` tasks are ever
        # outstanding, so requesting the full count would block waiting for outputs that are
        # gated behind downstream completion -> deadlock. One-at-a-time returns as soon as any
        # frame is split, so inference can be submitted immediately.
        while seen < args.count:
            for out in seg_sess.fetch_task_output(1, 30):
                seen += 1
                reply = _reply(out)
                image_id = reply.get("image_id")
                if not out.is_successful() or not reply.get("ok"):
                    with state_lock:
                        totals["seg_err"] += 1
                    _finish(image_id, True)
                    continue
                for tile in range(n_tiles):
                    tsa = soamapi.TaskSubmissionAttributes()
                    tsa.set_task_input(PipeMessage(
                        {"image_id": image_id, "tile": tile, "model": args.model}))
                    inf_sess.send_task_input(tsa)

    # stage 2 fetch -> stage 3 submit (one stitch task once all tiles are in)
    def inf_fetch():
        seen = 0
        target = args.count * n_tiles
        while seen < target:
            for out in inf_sess.fetch_task_output(1, 30):   # one at a time (see seg_fetch)
                seen += 1
                reply = _reply(out)
                image_id = reply.get("image_id")
                ok = out.is_successful() and reply.get("ok")
                with state_lock:
                    totals["inf_done"] += 1
                    if ok:
                        per_host[reply["host"]] += 1
                        per_host_us[reply["host"]] += reply.get("inference_us", 0)
                        per_host_id.setdefault(reply["host"],
                                               {"device": reply.get("device"),
                                                "product": reply.get("product"),
                                                "model": reply.get("model")})
                        totals["decode_us"] += reply.get("decode_us", 0)
                    else:
                        totals["inf_err"] += 1
                    state = img_state.get(image_id)
                    if state is None:
                        continue
                    state["got"] += 1
                    if not ok:
                        state["err"] = True
                    # dispatch the next stage exactly once, when every tile is in (a
                    # duplicate/retried inference output must not re-trigger it)
                    action = None
                    if state["got"] >= n_tiles and not state["dispatched"]:
                        state["dispatched"] = True
                        action = "fail" if state["err"] else "stitch"
                if action == "fail":
                    _finish(image_id, True)
                elif action == "stitch":
                    header = dict(stitch_header)
                    header["image_id"] = image_id
                    tsa = soamapi.TaskSubmissionAttributes()
                    tsa.set_task_input(PipeMessage(header))
                    stitch_sess.send_task_input(tsa)

    # stage 3 fetch -> record final detections, clean up, release the in-flight slot
    def stitch_fetch():
        # only frames whose tiles all succeeded reach this stage
        while not done_evt.is_set():
            for out in stitch_sess.fetch_task_output(1, 5):   # one at a time (see seg_fetch)
                reply = _reply(out)
                ok = out.is_successful() and reply.get("ok")
                image_id = reply.get("image_id")
                if ok:
                    with state_lock:
                        totals["boxes"] += len(reply.get("boxes") or [])
                        for name, count in (reply.get("class_hist") or {}).items():
                            classes[name] += count
                        sample = (img_state.get(image_id) or {}).get("sample")
                    if dump_file is not None:
                        record = {"sample": sample, "boxes": reply.get("boxes"),
                                  "scores": reply.get("scores"), "labels": reply.get("labels"),
                                  "truncated": reply.get("truncated")}
                        with dump_lock:
                            dump_file.write(json.dumps(record) + "\n")
                _finish(image_id, not ok)

    # watchdog: if no frame completes for --stall-timeout seconds the fleet has likely stalled
    # (e.g. a chip stuck under sustained load); abandon the in-flight frames and report partial
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

    started = time.time()
    with state_lock:
        progress["last"] = started
    threads = [threading.Thread(target=f)
               for f in (seg_submit, seg_fetch, inf_fetch, stitch_fetch, watchdog)]
    for thread in threads:
        thread.daemon = True
        thread.start()
    done_evt.wait()
    wall = time.time() - started

    # all frames finished -> the seg/inf fetch loops have already hit their targets; give the
    # stitch loop a moment to fall out of its poll before tearing the sessions down.
    for thread in threads:
        thread.join(timeout=8)
    if dump_file is not None:
        dump_file.close()

    for conn, sess in ((seg_conn, seg_sess), (inf_conn, inf_sess), (stitch_conn, stitch_sess)):
        try:
            sess.close(); conn.close()
        except Exception:
            pass
    soamapi.uninitialize()

    ok_frames = totals["img_done"] - totals["img_err"]
    tiles_ok = sum(per_host.values())
    avg_tile_ms = (sum(per_host_us.values()) / tiles_ok / 1000.0) if tiles_ok else 0.0
    rate = (args.count / wall) if wall else 0.0
    # a single chip runs a frame's tiles serially -> its frames/sec = 1 / (tiles * tile latency)
    one_chip = (1000.0 / (n_tiles * avg_tile_ms)) if avg_tile_ms else 0.0
    result = {
        "model": args.model, "count": args.count, "tiles": n_tiles,
        "input_source": pool.description, "is_random": pool.is_random,
        "sample_set": args.samples or args.model, "source_npz": pool.source,
        "ordered": args.ordered, "dump": dump_path,
        "post_thresh": (args.post_thresh if args.post_thresh is not None
                        else meta["post_merge_thresh"]),
        "stalled": stalled["v"],
        "images_done": ok_frames, "image_errors": totals["img_err"],
        "segments_done": tiles_ok, "segment_errors": totals["inf_err"] + totals["seg_err"],
        "chips": len(per_host), "wall_s": round(wall, 3),
        "throughput": round(rate, 2), "avg_seg_ms": round(avg_tile_ms, 3),
        "avg_decode_ms": round(totals["decode_us"] / tiles_ok / 1000.0, 3) if tiles_ok else 0.0,
        "one_chip_rate": round(one_chip, 2),
        "speedup": round(rate / one_chip, 2) if one_chip else 0.0,
        "avg_boxes": round(float(totals["boxes"]) / ok_frames, 2) if ok_frames else 0.0,
        "per_host": {host: dict(per_host_id.get(host, {}),
                                tasks=per_host[host],
                                avg_ms=round(per_host_us[host] / per_host[host] / 1000.0, 3))
                     for host in per_host},
        "classes": dict(classes),
    }
    if args.json:
        print(json.dumps(result))
        return

    print("\n=== shard pipeline across the Akida fleet ===")
    if stalled["v"]:
        print("!! STALLED: no frame completed for %.0fs (fleet likely stuck under load); "
              "partial results below. Tear down + relaunch (reset chips) and retry a smaller "
              "batch." % args.stall_timeout)
    print("input frames:  %s (%dx%dx%d, split into %d tiles)"
          % (pool.description, shape[0], shape[1], shape[2], n_tiles))
    if pool.is_random:
        print("               random noise contains no objects, so an empty result is the "
              "correct one; this run measures throughput, not accuracy.")
    print("chips used:    %d" % result["chips"])
    print("frames:        %d done, %d error" % (result["images_done"], result["image_errors"]))
    print("tiles:         %d done, %d error" % (result["segments_done"],
                                                result["segment_errors"]))
    print("wall time:     %.2f s" % wall)
    print("throughput:    %.2f frames/sec  (avg %.2f boxes/frame)" % (rate, result["avg_boxes"]))
    print("\nper-chip tile distribution:")
    for host in sorted(per_host):
        entry = result["per_host"][host]
        print("  %-26s %-9s %6d tiles   avg on-chip %.2f ms"
              % (host, entry.get("product") or "?", per_host[host], entry["avg_ms"]))
    print("\navg on-chip latency/tile:      %.2f ms (+ %.2f ms decode)"
          % (avg_tile_ms, result["avg_decode_ms"]))
    print("one chip (%d tiles serial):     ~%.2f frames/sec" % (n_tiles, one_chip))
    print("fleet of %d chips:              %.2f frames/sec  (~%.1fx a single chip)"
          % (len(per_host), rate, result["speedup"]))
    print("\nclass histogram: %s" % dict(classes))
    if dump_path:
        print("\ndetections written to %s" % dump_path)
        print("score them with: uv run python scripts/eval_shard_map.py --dump "
              ".cluster/shared/results/%s" % os.path.basename(dump_path))


def _tile_count(meta):
    """How many tiles the layout produces -- the client only needs the count, not the geometry.

    Kept as a tiny table rather than importing the geometry module: the client runs under the
    master's python3.6 soamapi binding, which has no numpy.
    """
    layout = meta["tile_layout"]
    return {"whole": 1, "quadrants_center": 5, "quadrants_global": 5,
            "quadrants_center_global": 6}[layout]


if __name__ == "__main__":
    main()
