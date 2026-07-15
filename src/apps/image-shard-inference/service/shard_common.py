"""Pure-stdlib shard/stitch geometry for the image-shard-inference pipeline (Python 3.6).

Shared by the Segment and Stitch SOAM service containers (both run under the soamapi 3.6
binding, which has no numpy) -- so everything here is plain Python: byte-slicing for the
shard, and small-list math for the YOLO decode + NMS stitch.

Layout: the app input is a 448x448x3 row-major uint8 image; it is split into five 224x224x3
segments (four quadrants + an overlapping center). The model output per segment is a flat
gy*gx*(A*(5+C)) int grid (C-order over [row][col][anchor*(box+class)]); stitching decodes each
grid, offsets boxes into the full-image frame, and NMS-merges across all five.
"""
import math

IMG_H = 448
IMG_W = 448
CH = 3
SEG = 224  # segment side

# (name, row0, col0) origin of each 224x224 segment within the 448 image.
SEGMENTS = [
    ("top_left",     0,   0),
    ("top_right",    0,   224),
    ("bottom_left",  224, 0),
    ("bottom_right", 224, 224),
    ("center",       112, 112),   # overlaps all four quadrants
]

# Decode tuning (cosmetic: inputs are random, accuracy is not evaluated). The model emits
# large int32 potentials; scale them into a sane logit range and clamp box exponents so the
# decoded boxes stay bounded and NMS is fast. On random noise the detector's objectness is
# strongly negative everywhere, so instead of a hard confidence gate (which would drop every
# box) we keep the top MAX_BOXES_PRE_NMS by score and NMS-merge them -- the stitch always
# surfaces a coordinate-remapped, merged result to demonstrate the stage.
LOGIT_SCALE = 1.0 / 8192.0
EXP_CLAMP = 2.0
SCORE_THRESHOLD = 0.0
MAX_BOXES_PRE_NMS = 200
IOU_THRESHOLD = 0.5


def shard(img, w=IMG_W, h=IMG_H, c=CH, s=SEG):
    """Split a row-major HxWxC uint8 byte image into the five SEGMENTS (each s*s*c bytes).

    Each segment row is a contiguous span of the source image, so we copy s row-slices.
    """
    rowbytes = s * c
    segs = []
    for _, r0, c0 in SEGMENTS:
        buf = bytearray(s * rowbytes)
        for rr in range(s):
            src = ((r0 + rr) * w + c0) * c
            buf[rr * rowbytes:(rr + 1) * rowbytes] = img[src:src + rowbytes]
        segs.append(bytes(buf))
    return segs


def _sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _softmax(xs):
    m = max(xs)
    es = [math.exp(v - m) for v in xs]
    s = sum(es) or 1.0
    return [e / s for e in es]


def _decode_grid(grid, gy, gx, num_anchors, box_per_anchor, anchors):
    """Decode one segment's flat int grid -> list of (x1,y1,x2,y2,score,cls) in seg pixels."""
    num_classes = box_per_anchor - 5
    cell = float(SEG) / gx
    boxes = []
    for row in range(gy):
        for col in range(gx):
            for a in range(num_anchors):
                base = ((row * gx + col) * num_anchors + a) * box_per_anchor
                tx = grid[base + 0] * LOGIT_SCALE
                ty = grid[base + 1] * LOGIT_SCALE
                tw = max(-EXP_CLAMP, min(EXP_CLAMP, grid[base + 2] * LOGIT_SCALE))
                th = max(-EXP_CLAMP, min(EXP_CLAMP, grid[base + 3] * LOGIT_SCALE))
                obj = _sigmoid(grid[base + 4] * LOGIT_SCALE)
                if num_classes > 1:
                    probs = _softmax([grid[base + 5 + k] * LOGIT_SCALE for k in range(num_classes)])
                    cls = max(range(num_classes), key=lambda k: probs[k])
                    score = obj * probs[cls]
                else:
                    cls, score = 0, obj
                cx = (col + _sigmoid(tx)) * cell
                cy = (row + _sigmoid(ty)) * cell
                bw = anchors[a][0] * math.exp(tw) * cell
                bh = anchors[a][1] * math.exp(th) * cell
                boxes.append((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2, score, cls))
    return boxes


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _nms(boxes, iou_thresh=IOU_THRESHOLD):
    """Per-class greedy NMS over (x1,y1,x2,y2,score,cls) tuples."""
    keep = []
    by_cls = {}
    for b in boxes:
        by_cls.setdefault(b[5], []).append(b)
    for cls in by_cls:
        cand = sorted(by_cls[cls], key=lambda x: x[4], reverse=True)
        while cand:
            best = cand.pop(0)
            keep.append(best)
            cand = [b for b in cand if _iou(best, b) < iou_thresh]
    return keep


def stitch(grids, meta):
    """Decode all segment grids, offset boxes into the 448 frame, NMS-merge.

    grids: list of 5 flat int lists (one per SEGMENTS entry). meta: model meta sidecar.
    Returns (detections, class_hist) where detection = dict(x1,y1,x2,y2,score,cls,cls_name).
    """
    gy, gx = meta.get("grid", [7, 7])
    num_anchors = meta.get("num_anchors", 5)
    box_per_anchor = meta.get("box_per_anchor", 7)
    anchors = meta.get("anchors")
    class_names = meta.get("class_names") or []

    merged = []
    for k, grid in enumerate(grids):
        if not grid:
            continue
        _, r0, c0 = SEGMENTS[k]
        for (x1, y1, x2, y2, score, cls) in _decode_grid(grid, gy, gx, num_anchors,
                                                          box_per_anchor, anchors):
            if score < SCORE_THRESHOLD:
                continue
            # offset segment-local pixels into the full 448 frame, then clip
            gx1 = min(IMG_W, max(0.0, x1 + c0))
            gy1 = min(IMG_H, max(0.0, y1 + r0))
            gx2 = min(IMG_W, max(0.0, x2 + c0))
            gy2 = min(IMG_H, max(0.0, y2 + r0))
            merged.append((gx1, gy1, gx2, gy2, score, cls))

    merged.sort(key=lambda b: b[4], reverse=True)
    merged = merged[:MAX_BOXES_PRE_NMS]
    kept = _nms(merged)

    dets, hist = [], {}
    for (x1, y1, x2, y2, score, cls) in kept:
        name = class_names[cls] if cls < len(class_names) else str(cls)
        dets.append({"x1": round(x1, 1), "y1": round(y1, 1), "x2": round(x2, 1),
                     "y2": round(y2, 1), "score": round(score, 4),
                     "cls": cls, "cls_name": name})
        hist[name] = hist.get(name, 0) + 1
    return dets, hist
