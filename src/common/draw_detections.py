"""Render a 448 frame with the fleet's detections drawn on it, as PNG bytes.

Host-side only (needs Pillow), used by the image-shard-inference dashboard gallery. Boxes
arrive frame-normalised, which is what the stitch stage returns, so drawing is a plain multiply
by the frame size -- no letterbox offset to undo, because the frame was produced by stretching
the source image to a square.

Ground truth, when shown, is in raw source pixels, so it goes the other way: divide by the raw
shape to normalise before scaling to the frame.
"""
import colorsys
import io

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# One colour per class. Hues step by 7/20 of the wheel rather than 1/20, so classes that are
# adjacent in the VOC list (and so likely to co-occur) land far apart in hue; saturation and
# value alternate to separate the two hues that inevitably come close. Kept bright enough to
# read against both dark and light photographs.
_PALETTE = [tuple(int(255 * c) for c in colorsys.hsv_to_rgb((i * 7 % 20) / 20.0,
                                                            0.85 if i % 2 else 0.65,
                                                            0.80 if i % 3 else 1.00))
            for i in range(20)]
_TRUTH = (255, 255, 255)
_FONTS = ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
          "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf")


def _font(size):
    for path in _FONTS:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _colour(label):
    return _PALETTE[int(label) % len(_PALETTE)]


def render(frame, boxes, scores, labels, class_names, truncated=None, truth=None,
           raw_shape=None, scale=1):
    """PNG bytes of `frame` with detections drawn.

    frame: (size, size, 3) uint8 RGB. boxes: (N, 4) xyxy normalised to the frame.
    truncated: optional (N,) bool; a seam still cuts these off, drawn dashed.
    truth: optional (gt_boxes in raw source pixels, gt_labels), drawn in white behind.
    raw_shape: (height, width) of the source image, required with truth.
    scale: integer upscale of the output, for a crisper gallery tile.
    """
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8))
    if scale != 1:
        image = image.resize((image.width * scale, image.height * scale), Image.NEAREST)
    size = image.width
    draw = ImageDraw.Draw(image, "RGBA")
    # Sized so the label stays legible when a 448 frame is shown at ~330px in the dashboard
    # gallery, without crowding out the picture in a frame carrying ten detections.
    label_font = _font(max(11, size // 30))

    if truth is not None:
        gt_boxes, gt_labels = truth
        height, width = int(raw_shape[0]), int(raw_shape[1])
        truth_font = _font(max(9, size // 44))
        for box, label in zip(np.asarray(gt_boxes, dtype=np.float64), gt_labels):
            x1, y1, x2, y2 = box / [width, height, width, height] * size
            draw.rectangle([x1, y1, x2, y2], outline=_TRUTH + (170,), width=max(1, size // 224))
            draw.text((x1 + 2, y1 + 1), class_names[int(label)], font=truth_font,
                      fill=_TRUTH + (200,))

    # Two passes, weakest score first, so a stronger detection's box never overdraws a weaker
    # one's label and every label ends up on top of every box.
    order = np.argsort(np.asarray(scores, dtype=np.float64))
    width_px = max(2, size // 150)
    drawn = []
    for i in order:
        x1, y1, x2, y2 = np.asarray(boxes[i], dtype=np.float64) * size
        colour = _colour(labels[i])
        cut = truncated is not None and bool(truncated[i])
        if cut:
            _dashed(draw, (x1, y1, x2, y2), colour, width_px)
        else:
            draw.rectangle([x1, y1, x2, y2], outline=colour + (255,), width=width_px)
        drawn.append((x1, y1, colour,
                      "%s %.2f%s" % (class_names[int(labels[i])], scores[i],
                                     " cut" if cut else "")))

    pad = 3
    for x1, y1, colour, text in drawn:
        left, top, right, bottom = draw.textbbox((0, 0), text, font=label_font)
        label_w, label_h = right - left + 2 * pad, bottom - top + 2 * pad
        # Keep the whole label inside the frame: a box hugging the right or top edge would
        # otherwise have its class name cut in half.
        x = min(max(0.0, x1), max(0.0, size - label_w))
        y = y1 - label_h if y1 - label_h >= 0 else min(y1, size - label_h)
        draw.rectangle([x, y, x + label_w, y + label_h], fill=colour + (235,))
        draw.text((x + pad, y + pad - top), text, font=label_font, fill=(16, 20, 32, 255))

    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def _dashed(draw, box, colour, width, dash=9):
    """A dashed rectangle, marking a detection a tile seam still cuts off."""
    x1, y1, x2, y2 = box
    for start, end, horizontal in ((x1, x2, True), (x1, x2, False)):
        position = start
        while position < end:
            stop = min(position + dash, end)
            y = y1 if horizontal else y2
            draw.line([position, y, stop, y], fill=colour + (255,), width=width)
            position += 2 * dash
    for start, end, left in ((y1, y2, True), (y1, y2, False)):
        position = start
        while position < end:
            stop = min(position + dash, end)
            x = x1 if left else x2
            draw.line([x, position, x, stop], fill=colour + (255,), width=width)
            position += 2 * dash
