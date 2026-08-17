"""Check the ported tiled pipeline box-for-box against the reference test kit.

Runs models/tiled_yolov2_voc.fbz on one Akida chip over the frames in a voc2007_test_r448
.npz and compares every merged detection against the reference detections stored in that same
file -- boxes, scores, labels *and* the truncated flag. Akida inference is deterministic and
the npz stores float32, so these are exact targets, not ranges.

This is the correctness gate for the whole app. It fails on the first frame that disagrees and
names the exact box, rather than showing up later as a vague point or two of mAP. If it passes
on every frame then the mAP is identical to the published one by construction.

Also asserts that models/tiled_yolov2_voc_meta.json agrees with the configuration the npz
carries -- anchors, labels, tile geometry, thresholds and all nine merge parameters -- which is
the cheapest guard against a mismatched model/anchors pair.

Runs inside the demo image (it needs akida). Use scripts/verify_reference.sh, which launches a
throwaway privileged container on one chip.

    scripts/verify_reference.sh --frames all
    scripts/verify_reference.sh --npz ~/data/voc/VOCdevkit/voc2007_test_r448.npz --frames all
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.environ.get("AKIDA_COMMON_DIR", "/opt/akida-common"))
from testkit import TestKit  # noqa: E402
from tiled_shard import (anchor_pixel_sizes, decode_tile, make_tile_layout,  # noqa: E402
                         merge_tile_detections, split_frame)

import akida  # noqa: E402


def check_meta(meta, npz):
    """Every field the reference pinned, cross-checked against the shipped meta."""
    merge = meta["merge"]
    checks = [
        ("class_names", meta["class_names"], [str(x) for x in npz["labels"]]),
        ("anchors", np.asarray(meta["anchors"], dtype=np.float32).tolist(),
         npz["anchors"].tolist()),
        ("frame_size", meta["frame_size"], int(npz["frame_size"])),
        ("input_size", meta["input_shape"][0], int(npz["input_size"])),
        ("tile_layout", meta["tile_layout"], str(npz["layout"])),
        ("obj_thresh", meta["obj_thresh"], float(npz["obj_thresh"])),
        ("nms_thresh", meta["nms_thresh"], float(npz["nms_thresh"])),
        ("max_boxes", meta["max_boxes"], int(npz["max_boxes"])),
    ]
    for name in ("nms_iou", "ios_thresh", "seam_iou", "edge_eps", "fuse_seams",
                 "prefer_complete", "only_truncated_victims", "clip", "truncated_penalty"):
        want = npz["merge_" + name]
        checks.append(("merge." + name, merge[name],
                       bool(want) if want.dtype == bool else float(want)))

    bad = [(n, got, want) for n, got, want in checks if got != want]
    for name, got, want in bad:
        print("  META MISMATCH %-26s meta=%r npz=%r" % (name, got, want))
    return not bad


def check_tiles(tiles, npz):
    names = [str(x) for x in npz["tile_names"]]
    origins = npz["tile_origins"].tolist()
    sizes = npz["tile_sizes"].tolist()
    ok = True
    for i, tile in enumerate(tiles):
        want = (names[i], origins[i][0], origins[i][1], sizes[i])
        got = (tile.name, tile.x0, tile.y0, tile.size)
        if got != want:
            print("  TILE MISMATCH index %d: %r != %r" % (i, got, want))
            ok = False
    if len(tiles) != len(names):
        print("  TILE COUNT %d != %d" % (len(tiles), len(names)))
        ok = False
    return ok


def report_frame(index, got, want):
    boxes, scores, labels, truncated = want
    print("  MISMATCH frame %d: %d merged boxes, reference has %d"
          % (index, len(got.boxes), len(boxes)))
    for i in range(max(len(got.boxes), len(boxes))):
        def show(b, s, lab, t, i):
            return ("%s s=%.6f l=%d t=%d" % (np.round(b[i], 6), s[i], lab[i], t[i])
                    if i < len(b) else "-")
        mine = show(got.boxes, got.scores, got.labels, got.truncated, i)
        theirs = show(boxes, scores, labels, truncated, i)
        print("   %s [%d] got  %s" % (" " if mine == theirs else "*", i, mine))
        print("     %s want %s" % (" " * len(str(i)), theirs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", default="/data/voc2007_test_r448_100.npz")
    parser.add_argument("--model", default="/models/tiled_yolov2_voc.fbz")
    parser.add_argument("--meta", default="/models/tiled_yolov2_voc_meta.json")
    parser.add_argument("--frames", default="500", help="how many frames, or 'all'")
    parser.add_argument("--software", action="store_true",
                        help="skip the hardware map and run the Akida software backend")
    args = parser.parse_args()

    kit = TestKit(args.npz)
    npz = kit
    meta = json.load(open(args.meta))
    num_classes = meta["num_classes"]
    input_size = meta["input_shape"][0]
    frame_size = meta["frame_size"]
    tiles = make_tile_layout(meta["tile_layout"], frame_size, input_size)

    print("model   %s" % args.model)
    print("data    %s (%s)" % (args.npz, str(npz["source_split"])))
    print("tiles   %s" % ", ".join("%d:%s@(%d,%d)/%d" % (i, t.name, t.x0, t.y0, t.size)
                                   for i, t in enumerate(tiles)))
    print("anchors %s px" % np.round(anchor_pixel_sizes(meta["anchors"]), 1).tolist())

    ok = check_meta(meta, npz) & check_tiles(tiles, npz)
    largest = anchor_pixel_sizes(meta["anchors"]).max()
    if largest > input_size:
        print("  ANCHOR MISMATCH largest anchor is %.0f px, larger than the %d px model input"
              % (largest, input_size))
        ok = False
    if not ok:
        print("\nFAILED: configuration does not match the reference; not running inference.")
        return 1
    print("config  matches the reference kit")

    model = akida.Model(args.model)
    if not args.software:
        devices = akida.devices()
        if not devices:
            print("\nFAILED: no Akida device visible.")
            return 1
        mode = getattr(akida.MapMode, meta.get("map_mode", "AllNps"))
        model.map(devices[0], hw_only=True, mode=mode)
        print("device  %s, mapped hw_only mode=%s, %d sequence(s)"
              % (getattr(devices[0], "desc", devices[0]), meta.get("map_mode"),
                 len(model.sequences)))
    else:
        print("device  none (software backend)")

    total = kit.count
    count = total if args.frames == "all" else min(int(args.frames), total)
    mismatches = 0
    started = time.time()
    print("\nchecking %d of %d frames..." % (count, total))
    for index in range(count):
        crops = split_frame(kit.frames[index], tiles, input_size)
        outputs = model.predict(crops)
        per_tile = [decode_tile(output, meta["anchors"], num_classes, meta["obj_thresh"],
                                meta["nms_thresh"]) for output in outputs]
        got = merge_tile_detections(per_tile, tiles, frame_size, max_boxes=meta["max_boxes"],
                                    **meta["merge"])
        want = kit.reference(index)
        same = all(np.array_equal(a, b) for a, b in
                   zip((got.boxes, got.scores, got.labels, got.truncated), want))
        if not same:
            mismatches += 1
            if mismatches <= 3:
                report_frame(index, got, want)
        if (index + 1) % 100 == 0:
            rate = (index + 1) / (time.time() - started)
            print("  %d/%d frames, %d mismatch(es), %.1f frames/s"
                  % (index + 1, count, mismatches, rate))

    elapsed = time.time() - started
    print("\n%d/%d frames match exactly (%.1f s, %.2f frames/s, %.1f ms per 6-tile frame)"
          % (count - mismatches, count, elapsed, count / elapsed, elapsed / count * 1000))
    if mismatches:
        print("FAILED: %d frame(s) disagree with the reference." % mismatches)
        return 1
    print("PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
