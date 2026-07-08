"""Generate a random sample set for an Akida model (host utility, run via uv).

Samples are random uint8 arrays matching the model's input shape -- they drive
load/throughput demos (the batch client generates the same thing inline via
--count; this writes a reusable file, e.g. a 5000-sample set to inspect or share).

    uv run python src/common/generate_samples.py --model kws_keyword_spotting --count 5000
"""
import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
MODELS_DIR = os.path.join(REPO, "models")
OUT_DIR = os.path.join(REPO, "samples")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--count", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    meta_path = os.path.join(MODELS_DIR, args.model + "_meta.json")
    if not os.path.isfile(meta_path):
        raise SystemExit("no metadata: %s" % meta_path)
    meta = json.load(open(meta_path))
    shape = meta["input_shape"]
    n = int(np.prod(shape))

    rng = np.random.default_rng(args.seed)
    samples = rng.integers(0, 256, size=(args.count, n), dtype=np.uint8).tolist()

    out = args.out or os.path.join(OUT_DIR, args.model + ".samples.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump({"model": args.model, "input_shape": shape,
                   "num_classes": meta.get("num_classes"),
                   "class_names": meta.get("class_names"),
                   "samples": samples}, fh)
    print("wrote %d samples (%d ints each) -> %s" % (args.count, n, out))


if __name__ == "__main__":
    main()
