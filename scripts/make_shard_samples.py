"""Generate random full-size input images for the image-shard-inference app.

The app's client sends 448x448x3 images to the SegmentService, which splits each into six
224x224x3 tiles. This set is the fallback that always travels with the repo, so a fresh clone
can demo the fleet without the VOC test kit: it exercises the whole pipeline and every
throughput number, but the images are uniform noise, so the detector correctly finds nothing
in them and accuracy is not evaluated. Point the launcher at a real .npz
(`scripts/launch/up.sh image-shard-inference --dataset <npz>`) for detections and mAP.

The launcher's prepare_samples.py flattens this into a raw .bin, the same numpy-free path the
vww/kws sample sets use.

    uv run python scripts/make_shard_samples.py            # -> data/samples/tiled_yolov2_voc.npz
    uv run python scripts/make_shard_samples.py --count 128 --seed 7
"""
import argparse
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=64, help="number of random images")
    ap.add_argument("--size", type=int, default=448, help="square image side (px)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default=os.path.join(REPO, "data", "samples", "tiled_yolov2_voc.npz"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    images = rng.integers(0, 256, size=(args.count, args.size, args.size, 3), dtype=np.uint8)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, images=images)
    mb = os.path.getsize(args.out) / 1e6
    print("wrote %s: %d x %dx%dx3 uint8 (%.1f MB)"
          % (args.out, args.count, args.size, args.size, mb))


if __name__ == "__main__":
    main()
