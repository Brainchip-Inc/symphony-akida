"""Tiled (sharded) YOLO inference: tile geometry, per-tile decode and detection merging.

A 448 frame is split into six 224 tiles that run independently, one per Akida device, and
the per-tile detections are mapped back into the frame and merged into one result. Ported
from `akida_models.detection.tiled_inference` (geometry + merge) and
`akida_models.detection.processing.decode_output` (per-tile decode), branch
`feature/tiled-yolov2-r448`. Numpy only -- no akida, no TensorFlow, no OpenCV -- so the
segment and stitch stages can import it on the management host, which has no chip.

Two properties of this file are load bearing, and both were measured rather than reasoned:

* **Tile order is semantic.** Fusion refuses to pair two fragments from the same tile, so a
  permuted order silently disables it. Index 5, the sixth tile, is the whole frame downscaled
  to the model input; dropping it costs 8.9 mAP50 and lands below a single-device run.
* **Anchors are never rescaled per tile.** A YOLO head encodes size as
  `anchor * exp(t) = pixels / stride`, so anchors are in units of 32 *input* pixels and are
  independent of what the input represents. Decoding gives `anchor * exp(t) / grid`, already
  the correct tile-normalised size, for a native-resolution crop and for a downscaled whole
  frame alike. Halving them for the sixth tile looks algebraically plausible and costs
  24.7 mAP50.

Boxes are always xyxy. Tile-local boxes are normalised to their own tile, merged boxes to the
full frame.
"""
from collections import namedtuple

import numpy as np

# Tolerance when deciding whether two tiles are truncated at the same seam. Seam positions are
# exact rationals in frame-normalised units, so this only guards against float noise.
_SEAM_TOL = 1e-6

LAYOUTS = ("quadrants_center", "quadrants_global", "quadrants_center_global", "whole")


class Tile(namedtuple("Tile", ["name", "x0", "y0", "size"])):
    """The square region [x0, x0+size) x [y0, y0+size) of the frame, processed as one tile.

    When size differs from the model input the tile is resized; otherwise it is a plain crop.
    """

    __slots__ = ()

    @property
    def x1(self):
        return self.x0 + self.size

    @property
    def y1(self):
        return self.y0 + self.size

    def seam_sides(self, frame_size):
        """Which sides of this tile are interior seams rather than true frame borders.

        A detection touching a frame border is legitimately cut off by the edge of the image.
        One touching an interior seam is a fragment of an object that continues into a
        neighbouring tile, and is a candidate for fusion.
        """
        return {"left": self.x0 > 0,
                "right": self.x1 < frame_size,
                "top": self.y0 > 0,
                "bottom": self.y1 < frame_size}


TileDetections = namedtuple("TileDetections",
                            ["boxes", "scores", "labels", "classes", "tile_ids", "truncated"])
TileDetections.__doc__ = """Detections as parallel arrays.

boxes (N,4) float xyxy; scores (N,) float; labels (N,) int; classes (N,num_classes) float or
None; tile_ids (N,) int, which tile each detection came from; truncated (N,) bool, True when a
tile seam still cuts the box off so it does not hold the object's full extent.
"""


def _empty_detections(num_classes=None):
    classes = None if num_classes is None else np.zeros((0, num_classes), dtype=np.float32)
    return TileDetections(boxes=np.zeros((0, 4), dtype=np.float32),
                          scores=np.zeros((0,), dtype=np.float32),
                          labels=np.zeros((0,), dtype=np.int64),
                          classes=classes,
                          tile_ids=np.zeros((0,), dtype=np.int64),
                          truncated=np.zeros((0,), dtype=bool))


def _index(detections, keep):
    """Applies an index or mask to every array of a TileDetections.

    Every stage below goes through this. `truncated` drives three separate mechanisms
    (prefer_complete, only_truncated_victims, truncated_penalty) worth ~5 mAP50 and 8.9 recall
    points between them, and it is invisible in the box coordinates, so it is the field a
    rewrite loses. Carrying it here means no stage can forget it.
    """
    keep = np.asarray(keep)
    return TileDetections(
        boxes=detections.boxes[keep],
        scores=detections.scores[keep],
        labels=detections.labels[keep],
        classes=None if detections.classes is None else detections.classes[keep],
        tile_ids=detections.tile_ids[keep],
        truncated=None if detections.truncated is None else detections.truncated[keep])


def make_tile_layout(layout, frame_size, tile_size):
    """Builds a tile layout for a frame.

    Quadrants sit flush against the frame corners, so they tile exactly when
    2*tile_size == frame_size and overlap when it is larger. `center` is centred, which for
    448/224 puts it at (112, 112) and makes it overlap a quarter of each quadrant. `global` is
    the whole frame, which the caller feeds to the model downscaled.

    For the shipped 448/224 model this returns, in order: top_left, top_right, bottom_left,
    bottom_right, center, global.
    """
    if layout not in LAYOUTS:
        raise ValueError("Unknown layout '%s', expected one of %s." % (layout, list(LAYOUTS)))
    if layout == "whole":
        return [Tile("whole", 0, 0, frame_size)]
    if not 0 < tile_size <= frame_size:
        raise ValueError("tile_size (%d) must be in ]0, frame_size (%d)]."
                         % (tile_size, frame_size))
    if 2 * tile_size < frame_size:
        raise ValueError("tile_size (%d) is too small to cover a frame of %d with four "
                         "quadrants: gaps would be left uncovered." % (tile_size, frame_size))

    step = frame_size - tile_size
    tiles = [Tile("top_left", 0, 0, tile_size),
             Tile("top_right", step, 0, tile_size),
             Tile("bottom_left", 0, step, tile_size),
             Tile("bottom_right", step, step, tile_size)]
    if layout in ("quadrants_center", "quadrants_center_global"):
        offset = (frame_size - tile_size) // 2
        tiles.append(Tile("center", offset, offset, tile_size))
    if layout in ("quadrants_global", "quadrants_center_global"):
        tiles.append(Tile("global", 0, 0, frame_size))
    return tiles


def _resize_down(crop, input_size):
    """Downscales a square uint8 crop by an exact integer factor, box-averaging each block.

    The reference uses cv2.resize(..., INTER_LINEAR). For an integer downscale the bilinear
    sample points land exactly between source pixels, so every output is the rounded mean of
    its k x k block -- verified bit-identical to OpenCV over 50 full VOC frames (0 differing
    pixels of 7,526,400) for the 448 -> 224 case this model uses. Doing it in numpy keeps the
    image free of opencv, whose wheel needs libGL and libxcb that the base image lacks.
    """
    size = crop.shape[0]
    k, remainder = divmod(size, input_size)
    if remainder:
        raise ValueError("Cannot box-average %d down to %d: not an integer factor."
                         % (size, input_size))
    if k == 1:
        return crop
    blocks = crop.reshape(input_size, k, input_size, k, -1).astype(np.uint32)
    total = blocks.sum(axis=(1, 3))
    half = (k * k) // 2
    return ((total + half) // (k * k)).astype(np.uint8).reshape(input_size, input_size, -1)


def split_frame(frame, tiles, input_size):
    """Cuts a frame into model-ready tiles, (len(tiles), input_size, input_size, channels)."""
    height, width = frame.shape[:2]
    if height != width:
        raise ValueError("Expected a square frame, got %dx%d." % (height, width))
    frame_size = height

    crops = np.empty((len(tiles), input_size, input_size) + frame.shape[2:], dtype=frame.dtype)
    for i, tile in enumerate(tiles):
        if tile.x0 < 0 or tile.y0 < 0 or tile.x1 > frame_size or tile.y1 > frame_size:
            raise ValueError("Tile %s falls outside a %dx%d frame." % (tile, frame_size,
                                                                       frame_size))
        crop = frame[tile.y0:tile.y1, tile.x0:tile.x1]
        if tile.size != input_size:
            crop = _resize_down(crop, input_size)
        crops[i] = crop
    return crops


def anchor_pixel_sizes(anchors, stride=32):
    """Anchors in grid-cell units -> the object pixel sizes they stand for.

    The cheapest guard against a mismatched model/anchors pair: every anchor should be a
    plausible fraction of the model input. The 448 checkpoint's anchors include a 311x336 box,
    which cannot fit in a 224 tile at all.
    """
    return np.asarray(anchors, dtype=np.float64) * stride


def tile_boxes_to_frame(boxes, tile, frame_size):
    """Maps tile-normalised xyxy boxes into frame-normalised xyxy boxes."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    scale = tile.size / frame_size
    offset = np.array([tile.x0, tile.y0, tile.x0, tile.y0], dtype=np.float32) / frame_size
    return boxes * scale + offset


def seam_truncations(boxes, tile, frame_size, edge_eps):
    """For each box, the edges at which an interior seam cuts it off.

    A box counts as truncated on a side when its edge sits within edge_eps of that side of the
    tile and the side is an interior seam rather than a frame border. decode_tile clamps the
    low edges at 0 but leaves the high edges free to overshoot, so the high-side test is
    one sided.

    Returns one set per box of (axis, edge) tuples: axis 0 is x and 1 is y, edge 0 is the box's
    low edge and 1 its high edge.
    """
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    seams = tile.seam_sides(frame_size)
    checks = []
    if seams["left"]:
        checks.append((0, 0))
    if seams["right"]:
        checks.append((0, 1))
    if seams["top"]:
        checks.append((1, 0))
    if seams["bottom"]:
        checks.append((1, 1))

    result = []
    for box in boxes:
        keys = set()
        for axis, edge in checks:
            if edge:
                touching = box[axis + 2] >= 1.0 - edge_eps
            else:
                touching = box[axis] <= edge_eps
            if touching:
                keys.add((axis, edge))
        result.append(keys)
    return result


def _pairwise_areas(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    low = np.maximum(a[:, None, :2], b[None, :, :2])
    high = np.minimum(a[:, None, 2:], b[None, :, 2:])
    side = np.clip(high - low, 0.0, None)
    intersection = side[..., 0] * side[..., 1]
    area_a = np.clip(a[:, 2] - a[:, 0], 0.0, None) * np.clip(a[:, 3] - a[:, 1], 0.0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0.0, None) * np.clip(b[:, 3] - b[:, 1], 0.0, None)
    return intersection, area_a, area_b


def iou_matrix(a, b):
    """(N, M) intersection over union between two sets of xyxy boxes."""
    intersection, area_a, area_b = _pairwise_areas(a, b)
    union = area_a[:, None] + area_b[None, :] - intersection
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, intersection / union, 0.0)


def ios_matrix(a, b):
    """(N, M) fraction of b[j] that lies inside a[i].

    This is what catches a truncated fragment sitting inside a complete detection: a fragment
    covering a third of an object has an IoU of only 0.33 with it, but an IoS of 1.0.
    """
    intersection, _, area_b = _pairwise_areas(a, b)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(area_b[None, :] > 0, intersection / area_b[None, :], 0.0)


def _greedy_suppress(detections, threshold, score_matrix_fn, prefer_complete=False,
                     only_truncated_victims=False):
    """Greedy per-class suppression driven by an arbitrary pairwise overlap measure.

    Candidates are normally visited in decreasing score. With prefer_complete, boxes that no
    seam cuts off are visited first, so a complete detection always outranks a truncated
    fragment it overlaps however confident that fragment is -- a fragment holds only part of an
    object, so letting it survive in place of the whole box is never right.

    With only_truncated_victims, only truncated boxes can be suppressed. That matters for
    containment: a small object genuinely sitting inside a larger one of the same class, a
    person in a crowd or a car among cars, is fully contained yet entirely real, and
    suppressing on containment alone cost 8.9 recall points, almost all on small objects.
    """
    num = len(detections.boxes)
    if num == 0:
        return detections
    have_truncation = detections.truncated is not None
    if prefer_complete and have_truncation:
        # lexsort takes the last key as primary, so complete boxes lead, score breaks ties.
        order = np.lexsort((-detections.scores, detections.truncated.astype(np.int8)))
    else:
        order = np.argsort(-detections.scores, kind="stable")
    overlap = score_matrix_fn(detections.boxes, detections.boxes)
    same_label = detections.labels[:, None] == detections.labels[None, :]
    eligible = (detections.truncated if only_truncated_victims and have_truncation
                else np.ones(num, dtype=bool))

    keep = []
    suppressed = np.zeros(num, dtype=bool)
    for idx in order:
        if suppressed[idx]:
            continue
        keep.append(idx)
        victims = same_label[idx] & (overlap[idx] >= threshold) & eligible
        victims[idx] = False
        suppressed |= victims
    return _index(detections, np.array(sorted(keep), dtype=np.int64))


def per_class_nms(detections, iou_threshold, prefer_complete=False):
    """Greedy per-class non-maximum suppression, in the detections' original order."""
    return _greedy_suppress(detections, iou_threshold, iou_matrix, prefer_complete)


def suppress_contained(detections, ios_threshold, prefer_complete=False,
                       only_truncated_victims=True):
    """Drops truncated fragments largely contained inside another box of the same class."""
    return _greedy_suppress(detections, ios_threshold, ios_matrix, prefer_complete,
                            only_truncated_victims)


def _projection_iou(box_a, box_b, axis):
    """1D IoU of two boxes projected onto one axis (0 for x, 1 for y)."""
    low = max(box_a[axis], box_b[axis])
    high = min(box_a[axis + 2], box_b[axis + 2])
    overlap = max(high - low, 0.0)
    extent_a = max(box_a[axis + 2] - box_a[axis], 0.0)
    extent_b = max(box_b[axis + 2] - box_b[axis], 0.0)
    union = extent_a + extent_b - overlap
    return overlap / union if union > 0 else 0.0


def _fragments_join(box_low, box_high, axis, seam_iou):
    """Whether two complementarily truncated boxes are two halves of one object.

    box_low is cut off at its high edge along axis and box_high at its low edge, so to be
    halves of one object box_low must sit before box_high along that axis and the two must meet
    rather than be far apart.
    """
    # Consistent ordering along the split axis.
    if box_low[axis] > box_high[axis] + _SEAM_TOL:
        return False
    if box_low[axis + 2] > box_high[axis + 2] + _SEAM_TOL:
        return False
    # The two halves have to actually meet, otherwise they are separate objects.
    if box_low[axis + 2] < box_high[axis] - _SEAM_TOL:
        return False
    # And line up across the split, which also rejects pairs of very different extents.
    return _projection_iou(box_low, box_high, 1 - axis) >= seam_iou


def fuse_seam_fragments(detections, seams, seam_iou):
    """Rebuilds objects that tile seams cut into pieces.

    Two detections fuse when they carry the same class, come from *different* tiles, and are
    complementarily truncated: along one axis one is cut off at its high edge while the other
    is cut off at its low edge, they are ordered consistently along that axis, they meet, and
    they line up across it. The pair is replaced by its enclosing box, on both axes. Fusion is
    transitive via union-find, so an object split across three or more tiles collapses in one
    pass, and a group of two or more counts as no longer truncated.

    The enclosing box is right on the axis *across* the split too, which is not obvious. It is
    tempting to read the two fragments' extents on that axis as two estimates of one edge and
    average them, but a seam cuts the object, not just its box: each fragment's extent across
    the split is the extent of its own piece, so a leaning person cut at the waist gives an
    upper fragment spanning only the head's rows. Averaging was tried and reverted; it costs
    2.9 mAP75.

    Matching deliberately does not require the two fragments to be cut by the *same* seam. With
    a centre tile overlapping the quadrants, a large object is typically cut at the quadrant
    seam in one tile and at the centre tile's own seam in another, at different positions;
    those are still two halves of one object and must fuse.
    """
    num = len(detections.boxes)
    detections = detections._replace(
        truncated=np.array([bool(keys) for keys in seams], dtype=bool))
    if num < 2:
        return detections

    parent = list(range(num))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(num):
        if not seams[i]:
            continue
        for j in range(i + 1, num):
            if not seams[j]:
                continue
            if detections.labels[i] != detections.labels[j]:
                continue
            if detections.tile_ids[i] == detections.tile_ids[j]:
                continue
            for axis in (0, 1):
                # i truncated at its high edge and j at its low edge, or the other way round.
                for low, high in ((i, j), (j, i)):
                    if (axis, 1) not in seams[low] or (axis, 0) not in seams[high]:
                        continue
                    if _fragments_join(detections.boxes[low], detections.boxes[high], axis,
                                       seam_iou):
                        union(i, j)

    groups = {}
    for i in range(num):
        groups.setdefault(find(i), []).append(i)
    if len(groups) == num:
        return detections

    boxes, scores, labels, classes, tile_ids, truncated = [], [], [], [], [], []
    for members in (groups[root] for root in sorted(groups)):
        member_boxes = detections.boxes[members]
        best = members[int(np.argmax(detections.scores[members]))]
        boxes.append(np.concatenate([member_boxes[:, :2].min(axis=0),
                                     member_boxes[:, 2:].max(axis=0)]))
        scores.append(detections.scores[members].max())
        labels.append(detections.labels[best])
        tile_ids.append(detections.tile_ids[best])
        truncated.append(len(members) == 1 and bool(seams[members[0]]))
        if detections.classes is not None:
            classes.append(detections.classes[members].max(axis=0))

    return TileDetections(
        boxes=np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
        scores=np.asarray(scores, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        classes=None if detections.classes is None else np.asarray(classes, dtype=np.float32),
        tile_ids=np.asarray(tile_ids, dtype=np.int64),
        truncated=np.asarray(truncated, dtype=bool))


def merge_tile_detections(per_tile, tiles, frame_size, nms_iou=0.5, ios_thresh=0.7,
                          seam_iou=0.2, edge_eps=0.05, fuse_seams=True, prefer_complete=True,
                          only_truncated_victims=True, clip=True, max_boxes=None,
                          truncated_penalty=0.4):
    """Merges per-tile detections into one frame-level result, sorted by decreasing score.

    In order: map every tile-local box into frame-normalised coordinates recording which
    interior seams cut it off; fuse fragments of one object split across a seam into their
    enclosing box; suppress boxes largely contained in another of the same class; per-class
    NMS to remove the ordinary duplicates from overlapping tiles; then clip, demote the
    fragments fusion could not complete, and keep the highest scoring max_boxes.

    This whole function is the difference between 22.70 and 49.14 mAP50 -- pooling the tiles
    and running one global per-class NMS instead is worth -26.4. The defaults are the tuned
    values, fitted on VOC trainval and never on a reporting split.

    per_tile: one (boxes, scores, labels, classes) tuple per tile, in the same order as tiles,
    boxes (N,4) xyxy normalised to the tile, classes optionally None. Entries may be empty.

    truncated_penalty scales the score of any detection a seam still cuts off, before ranking.
    Training makes a fragment with enough of an object visible a full-confidence positive, so
    the model is as sure about half an object as a whole one: lone fragments average 0.838
    against 0.872 for complete detections while reaching a mean IoU of only 0.231 against
    0.483. Their confidence is honest about "a fragment is here" and wrong about "a well
    localised object is here", and only the merge knows which fragments it failed to complete.
    Worth 4.5 mAP50; pass 1.0 to disable.
    """
    if len(per_tile) != len(tiles):
        raise ValueError("Got %d detection sets for %d tiles." % (len(per_tile), len(tiles)))

    all_boxes, all_scores, all_labels, all_classes, all_tiles, all_seams = [], [], [], [], [], []
    num_classes = None
    for tile_id, (tile, entry) in enumerate(zip(tiles, per_tile)):
        boxes, scores, labels, classes = entry
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        if classes is not None:
            classes = np.asarray(classes, dtype=np.float32)
            if classes.ndim != 2:
                classes = classes.reshape(len(boxes), -1)
            # Read the class count even from an empty tile, so a frame with no detections at
            # all still reports a correctly shaped, empty class array.
            num_classes = classes.shape[1]
        if len(boxes) == 0:
            continue
        all_seams.extend(seam_truncations(boxes, tile, frame_size, edge_eps))
        all_boxes.append(tile_boxes_to_frame(boxes, tile, frame_size))
        all_scores.append(np.asarray(scores, dtype=np.float32).reshape(-1))
        all_labels.append(np.asarray(labels, dtype=np.int64).reshape(-1))
        all_tiles.append(np.full(len(boxes), tile_id, dtype=np.int64))
        if classes is not None:
            all_classes.append(classes)

    if not all_boxes:
        return _empty_detections(num_classes)

    detections = TileDetections(
        boxes=np.concatenate(all_boxes),
        scores=np.concatenate(all_scores),
        labels=np.concatenate(all_labels),
        classes=np.concatenate(all_classes) if all_classes else None,
        tile_ids=np.concatenate(all_tiles),
        truncated=np.array([bool(keys) for keys in all_seams], dtype=bool))

    if fuse_seams:
        detections = fuse_seam_fragments(detections, all_seams, seam_iou)
    if ios_thresh <= 1.0:
        detections = suppress_contained(detections, ios_thresh, prefer_complete,
                                        only_truncated_victims)
    detections = per_class_nms(detections, nms_iou, prefer_complete)

    if clip:
        detections = detections._replace(boxes=np.clip(detections.boxes, 0.0, 1.0))
        widths = detections.boxes[:, 2] - detections.boxes[:, 0]
        heights = detections.boxes[:, 3] - detections.boxes[:, 1]
        detections = _index(detections, (widths > 0) & (heights > 0))

    if truncated_penalty != 1.0 and detections.truncated is not None:
        scale = np.where(detections.truncated, truncated_penalty, 1.0).astype(np.float32)
        detections = detections._replace(scores=detections.scores * scale)

    order = np.argsort(-detections.scores, kind="stable")
    if max_boxes is not None:
        order = order[:max_boxes]
    return _index(detections, order)


def _sigmoid(x):
    return 1. / (1. + np.exp(-x))


def _softmax(x, axis=-1, t=-100.):
    """decode_output's softmax, quirks included.

    It subtracts a *global* max rather than a per-row one, which is harmless since softmax is
    shift invariant per row, and then applies a rescale when some logit sits more than 100
    below that max, which is not. Both are reproduced verbatim: this decode has to agree with
    the reference box for box, and a clean textbook softmax disagrees in the third decimal
    whenever that branch fires.
    """
    x = x - np.max(x)
    if np.min(x) < t:
        x = x / np.min(x) * t
    e_x = np.exp(x)
    return e_x / e_x.sum(axis, keepdims=True)


def decode_tile(output, anchors, num_classes, obj_threshold=0.5, nms_threshold=0.5):
    """Decodes one tile's raw model output into tile-normalised detections.

    Port of `akida_models.detection.processing.decode_output`, returning parallel numpy arrays
    (boxes (N,4) xyxy, scores (N,), labels (N,), classes (N,num_classes)) instead of
    BoundingBox objects.

    output: (grid_h, grid_w, num_anchors*(5+num_classes)) as `akida.Model.predict` returns it,
    or already reshaped to (grid_h, grid_w, num_anchors, 5+num_classes). The flat channel axis
    is anchor major (channel = anchor_index*(5+num_classes) + attribute_index); reading it
    attribute major gives boxes that look almost plausible.

    Three details are load bearing. The high box edges are clamped at grid_w, *not* at 1.0, so
    a box may legitimately extend past the tile -- the merge's seam detection tests
    `x2 >= 1 - edge_eps`, and clamping at 1.0 would make every large object look truncated on
    every side. `score` is the objectness, not the class probability; the class scores only
    pick the label. And boxes stay tile local: the merge maps them to the frame, because it
    needs the tile-local box to decide what a seam cut.
    """
    anchors = np.asarray(anchors, dtype=np.float32)
    grid_h, grid_w = output.shape[:2]
    # decode writes into its input, so work on a private float copy.
    output = np.array(output, dtype=np.float32).reshape(grid_h, grid_w, len(anchors),
                                                        5 + num_classes)

    output[..., 4] = _sigmoid(output[..., 4])
    output[..., 5:] = output[..., 4][..., np.newaxis] * _softmax(output[..., 5:])
    output[..., 5:] *= output[..., 5:] > obj_threshold

    col, row, _ = np.meshgrid(np.arange(grid_w), np.arange(grid_h), np.arange(len(anchors)))

    x = (col + _sigmoid(output[..., 0])) / grid_w
    y = (row + _sigmoid(output[..., 1])) / grid_h
    w = anchors[:, 0] * np.exp(output[..., 2]) / grid_w
    h = anchors[:, 1] * np.exp(output[..., 3]) / grid_h

    x1 = np.maximum(x - w / 2, 0)
    y1 = np.maximum(y - h / 2, 0)
    x2 = np.minimum(x + w / 2, grid_w)
    y2 = np.minimum(y + h / 2, grid_h)

    confidence = output[..., 4]
    class_scores = output[..., 5:]
    keep = np.sum(class_scores, axis=-1) > 0
    rows, cols, boxes_idx = np.where(keep)
    if len(rows) == 0:
        return (np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int64), np.zeros((0, num_classes), dtype=np.float32))

    idx = (rows, cols, boxes_idx)
    boxes = np.stack([x1[idx], y1[idx], x2[idx], y2[idx]], axis=1).astype(np.float32)
    scores = confidence[idx].astype(np.float32)
    classes = class_scores[idx].astype(np.float32)
    labels = np.argmax(classes, axis=1).astype(np.int64)

    # Per-class NMS, exactly as decode_output does it: for each class, walk the boxes in
    # decreasing per-class score and zero the score of any later box overlapping too much --
    # but only when BOTH boxes' argmax label is that class.
    scores = scores.copy()
    overlaps = iou_matrix(boxes, boxes)
    for c in range(num_classes):
        order = np.argsort(classes[:, c])[::-1]
        for rank, i in enumerate(order):
            if scores[i] == 0 or classes[i, c] == 0:
                continue
            for j in order[rank + 1:]:
                if scores[j] == 0:
                    continue
                if overlaps[i, j] >= nms_threshold and labels[i] == c and labels[j] == c:
                    scores[j] = 0

    surviving = scores > obj_threshold
    return boxes[surviving], scores[surviving], labels[surviving], classes[surviving]
