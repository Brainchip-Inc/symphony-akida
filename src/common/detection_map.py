"""VOC-style mean average precision, ported from `akida_models.detection.map_evaluation`.

All-points AP (the py-faster-rcnn formulation) over IoU thresholds 0.50 to 0.95 in steps of
0.05, per class, with a per-image cap on detections. Kept byte-for-byte comparable to the
reference so the numbers this repo prints can be read against the published ones.

Numpy only: imported by the host-side dashboard and by scripts/eval_shard_map.py, both of
which run under uv rather than in the container.

Boxes are xyxy in *raw source pixels* on both sides. Scaling merged detections by the frame
size (448) instead of by each image's own raw shape silently changes every AP: the 448 frames
were produced by stretching each source image to a square, so the inverse is anisotropic and
IoU is not invariant under it.
"""
import numpy as np

IOU_THRESHOLDS = np.linspace(0.5, 0.95, num=10)


def compute_ap(recall, precision):
    """Average precision under the precision envelope, from py-faster-rcnn."""
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    changes = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[changes + 1] - mrec[changes]) * mpre[changes + 1]))


def compute_overlap(a, b):
    """(N, M) IoU between two sets of xyxy boxes; extra columns beyond 4 are ignored."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    width = np.minimum(a[:, 2:3], b[:, 2]) - np.maximum(a[:, 0:1], b[:, 0])
    height = np.minimum(a[:, 3:4], b[:, 3]) - np.maximum(a[:, 1:2], b[:, 1])
    intersection = np.maximum(width, 0) * np.maximum(height, 0)
    area_a = (a[:, 2:3] - a[:, 0:1]) * (a[:, 3:4] - a[:, 1:2])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return intersection / np.maximum(area_a + area_b - intersection, np.finfo(float).eps)


def group_by_label(boxes, labels, num_classes, extra=None):
    """Splits (N, 4) boxes into one array per class, optionally appending a score column."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    labels = np.asarray(labels).reshape(-1)
    if extra is not None:
        boxes = np.column_stack([boxes, np.asarray(extra, dtype=np.float64).reshape(-1)])
    return [boxes[labels == c] for c in range(num_classes)]


def average_precisions(all_detections, all_annotations, num_classes, iou_threshold, overlaps):
    """Per-class AP at one IoU threshold.

    all_detections[i][c] is (N, 5) xyxy + score, already sorted by decreasing score and capped;
    all_annotations[i][c] is (M, 4) xyxy. A detection matches the annotation it overlaps most,
    once each: a second detection on an already-matched annotation is a false positive.
    """
    result = []
    for label in range(num_classes):
        false_positives, true_positives, scores = [], [], []
        num_annotations = 0.0
        for i in range(len(all_detections)):
            detections = all_detections[i][label]
            annotations = all_annotations[i][label]
            num_annotations += annotations.shape[0]
            overlap = overlaps[(i, label)]
            detected = set()
            for idx, detection in enumerate(detections):
                scores.append(detection[4])
                if annotations.shape[0] == 0:
                    false_positives.append(1)
                    true_positives.append(0)
                    continue
                assigned = int(np.argmax(overlap, axis=1)[idx])
                if overlap[idx, assigned] >= iou_threshold and assigned not in detected:
                    false_positives.append(0)
                    true_positives.append(1)
                    detected.add(assigned)
                else:
                    false_positives.append(1)
                    true_positives.append(0)

        if num_annotations == 0:
            result.append(0.0)
            continue
        order = np.argsort(-np.asarray(scores))
        cumulative_tp = np.cumsum(np.asarray(true_positives)[order])
        cumulative_fp = np.cumsum(np.asarray(false_positives)[order])
        recall = cumulative_tp / num_annotations
        precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp,
                                               np.finfo(np.float64).eps)
        result.append(compute_ap(recall, precision))
    return result


def evaluate(all_detections, all_annotations, num_classes):
    """mAP over IOU_THRESHOLDS.

    Returns (per_threshold, per_class_ap) where per_threshold maps each IoU threshold to its
    mAP and per_class_ap is the per-class mean over all ten thresholds.
    """
    overlaps = {}
    for label in range(num_classes):
        for i in range(len(all_detections)):
            overlaps[(i, label)] = compute_overlap(all_detections[i][label],
                                                   all_annotations[i][label])
    per_threshold = {}
    per_class = np.zeros(num_classes)
    for threshold in IOU_THRESHOLDS:
        aps = average_precisions(all_detections, all_annotations, num_classes, threshold,
                                 overlaps)
        per_threshold[float(threshold)] = float(np.mean(aps))
        per_class += np.asarray(aps) / len(IOU_THRESHOLDS)
    return per_threshold, per_class


def summarise(per_threshold):
    """The three headline numbers: mAP50, mAP75 and the mean over IoU 0.50:0.95."""
    keys = sorted(per_threshold)
    return {"map50": per_threshold[keys[0]],
            "map75": per_threshold[keys[5]],
            "map": float(np.mean([per_threshold[k] for k in keys]))}
