"""Score a shard-pipeline detection dump against the VOC2007 test ground truth.

Reads the JSONL the client writes with --dump, pairs each record with its source frame's
ground truth, and reports mAP50, mAP75 and mAP over IoU 0.50:0.95.

Two rows come out, and the second one is the point of the design:

  fleet       what the six chips actually produced this run
  reference   the test kit's own stored detections, scored by the *same* code

The kit's detections are the published 49.14 mAP50 result, so if the fleet row and the
reference row agree, the pipeline is exact -- independently of whether this scorer reproduces
the published number to the last decimal. Any drift in the scorer moves both rows together
and cannot be mistaken for a pipeline regression.

That matters here, because the reference row lands a hair above the kit's stored targets:
+8.1e-5 mAP50, +1.6e-3 mAP75, +4.9e-4 mAP. This code is not the cause. Driving the reference
MapEvaluation's own _compute_all_overlaps and _calc_avg_precisions over the same arrays
reproduces this scorer bit for bit, and the overlap matrices are identical to the last ulp,
with zero TP/FP decisions flipping at either threshold; detection ordering makes no difference
either (stored, stable, and both quicksort dtypes all agree). The offset therefore comes from
how the stored targets were generated upstream, not from scoring, and it applies equally to
both rows.

    uv run python scripts/eval_shard_map.py
    uv run python scripts/eval_shard_map.py --dump .cluster/shared/results/<run>.jsonl \\
        --npz ~/data/voc/VOCdevkit/voc2007_test_r448.npz
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(REPO, "src", "common"))
from detection_map import evaluate, group_by_label, summarise  # noqa: E402
from testkit import TestKit, scale_to_raw  # noqa: E402

RESULTS_DIR = os.path.join(REPO, ".cluster", "shared", "results")
SAMPLES_DIR = os.path.join(REPO, ".cluster", "shared", "samples")


def newest_dump():
    dumps = sorted(glob.glob(os.path.join(RESULTS_DIR, "*.jsonl")), key=os.path.getmtime)
    if not dumps:
        raise SystemExit("no detection dump in %s; run the client with --dump" % RESULTS_DIR)
    return dumps[-1]


def find_source_npz(dump_path, needed_frames):
    """The .npz a dump was produced from, via the sample set sidecars the launcher wrote.

    The client names a dump after its sample set, so the set name is the basename's prefix.
    Falling back to "any kit with ground truth" would silently pick the wrong one when both the
    500-frame and the full split are prepared, so the fallback also has to be big enough to
    contain every sample the dump references.
    """
    basename = os.path.basename(dump_path)
    candidates = []
    for sidecar in glob.glob(os.path.join(SAMPLES_DIR, "*.samples.json")):
        side = json.load(open(sidecar))
        if not side.get("has_ground_truth") or not os.path.isfile(side.get("source_npz") or ""):
            continue
        name = side.get("set") or ""
        candidates.append((basename.startswith(name + "_"), int(side.get("count", 0)),
                           side["source_npz"]))
    named = [c for c in candidates if c[0]]
    big_enough = [c for c in candidates if c[1] >= needed_frames]
    for pool in (named, big_enough):
        if pool:
            return min(pool, key=lambda c: c[1])[2]
    raise SystemExit("could not find a prepared sample set with ground truth covering %d "
                     "frames; pass --npz (dump was %s)" % (needed_frames, dump_path))


def read_dump(path):
    records = {}
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            records[int(record["sample"])] = record
    if not records:
        raise SystemExit("%s is empty" % path)
    return records


def collect(kit, samples, boxes_of, max_boxes):
    """(detections, annotations) per frame, per class, in raw source pixels."""
    num_classes = len(kit.labels)
    detections, annotations = [], []
    for sample in samples:
        boxes, scores, labels = boxes_of(sample)
        if len(boxes):
            boxes = scale_to_raw(boxes, kit.raw_shapes[sample])
            order = np.argsort(-np.asarray(scores))[:max_boxes]
            boxes, scores, labels = boxes[order], np.asarray(scores)[order], \
                np.asarray(labels)[order]
        else:
            boxes = np.zeros((0, 4))
            scores, labels = np.zeros((0,)), np.zeros((0,), dtype=int)
        detections.append(group_by_label(boxes, labels, num_classes, extra=scores))
        gt_boxes, gt_labels = kit.annotations(sample)
        annotations.append(group_by_label(gt_boxes, gt_labels, num_classes))
    return detections, annotations


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=None, help="client JSONL (default: newest under .cluster)")
    ap.add_argument("--npz", default=None, help="source test kit (default: from the sidecar)")
    ap.add_argument("--max-boxes", type=int, default=None,
                    help="detections per frame to score (default: the kit's own max_boxes)")
    ap.add_argument("--post-thresh", type=float, default=0.0,
                    help="gate the reference row the same way the run was gated, so the two "
                         "rows stay comparable. Match whatever --post-thresh the client used; "
                         "the published figures assume 0")
    ap.add_argument("--per-class", action="store_true", help="also print per-class AP")
    ap.add_argument("--json", action="store_true", help="emit one JSON result line (dashboard)")
    args = ap.parse_args()

    dump_path = args.dump or newest_dump()
    records = read_dump(dump_path)
    samples = sorted(records)
    kit = TestKit(args.npz or find_source_npz(dump_path, samples[-1] + 1))
    if not kit.has_ground_truth:
        raise SystemExit("%s carries no ground truth" % kit.path)
    max_boxes = args.max_boxes or int(kit["max_boxes"])
    log_to = sys.stderr if args.json else sys.stdout

    if max(samples) >= kit.count:
        raise SystemExit("dump references sample %d but the kit has %d frames; was the client "
                         "run against a different set?" % (max(samples), kit.count))

    print("dump      %s (%d frames)" % (dump_path, len(samples)), file=log_to)
    print("test kit  %s (%d frames, %d classes)" % (kit.path, kit.count, len(kit.labels)),
          file=log_to)
    print("scoring   top %d detections per frame, IoU 0.50:0.95\n" % max_boxes, file=log_to)

    def from_dump(sample):
        record = records[sample]
        return record["boxes"], record["scores"], record["labels"]

    def from_kit(sample):
        boxes, scores, labels, _truncated = kit.reference(sample)
        # The reference carries every merged box, so it has to be gated exactly as the run was
        # or the two rows measure different things. Its scores already carry the truncated
        # penalty, so the same threshold means the same thing on both sides.
        keep = np.asarray(scores) >= args.post_thresh
        return np.asarray(boxes)[keep], np.asarray(scores)[keep], np.asarray(labels)[keep]

    rows = {}
    per_threshold, per_class = evaluate(*collect(kit, samples, from_dump, max_boxes),
                                        num_classes=len(kit.labels))
    rows["fleet"] = summarise(per_threshold)

    if kit.has_reference:
        ref_thresholds, _ = evaluate(*collect(kit, samples, from_kit, max_boxes),
                                     num_classes=len(kit.labels))
        rows["reference"] = summarise(ref_thresholds)

    targets = kit.targets() if len(samples) == kit.count else None
    result = {"dump": dump_path, "npz": kit.path, "frames": len(samples),
              "max_boxes": max_boxes, "targets": targets,
              "per_class": {name: round(float(ap), 6)
                            for name, ap in zip(kit.labels, per_class)}}
    result.update(rows["fleet"])
    result["reference"] = rows.get("reference")

    if args.json:
        print(json.dumps(result))
        return

    print("%-11s %9s %9s %9s" % ("", "mAP50", "mAP75", "mAP"))
    for name in ("fleet", "reference"):
        if name in rows:
            row = rows[name]
            print("%-11s %9.4f %9.4f %9.4f" % (name, row["map50"], row["map75"], row["map"]))
    if targets:
        print("%-11s %9.4f %9.4f %9.4f   (published, whole split)"
              % ("target", targets["map50"], targets["map75"], targets["map"]))
    if "reference" in rows:
        delta = max(abs(rows["fleet"][k] - rows["reference"][k])
                    for k in ("map50", "map75", "map"))
        verdict = "identical to the reference detections" if delta < 1e-9 \
            else "differs from the reference detections by %.2e" % delta
        print("\nfleet output is %s" % verdict)
        if targets:
            offset = max(abs(rows["reference"][k] - targets[k])
                         for k in ("map50", "map75", "map"))
            if offset >= 1e-6:
                print("the reference row itself sits %.1e off the published target, so read "
                      "the fleet row against it rather than against the target" % offset)
    elif not kit.has_reference:
        print("\n(this kit carries no reference detections to compare against)")

    if args.per_class:
        print("\nper-class AP (mean over IoU 0.50:0.95):")
        for name, value in sorted(zip(kit.labels, per_class), key=lambda kv: -kv[1]):
            print("  %-14s %.4f" % (name, value))


if __name__ == "__main__":
    main()
