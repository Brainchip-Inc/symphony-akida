"""Convert real .npz sample sets into a numpy-free form the SOAM client can serve.

The batch client runs under the master's Python 3.6 soamapi binding, which has no
numpy -- so it cannot read .npz directly. This host utility (run via uv, which has
numpy) pre-flattens each dataset into:

  <model>.bin           raw concatenated uint8, C-order, count * per_sample_bytes
  <model>.samples.json  {model, count, per_sample_bytes, input_shape, num_classes, class_names}

The client then mmap/slices the .bin into ready-to-send byte tensors (no parsing) and
ships them as binary. Datasets are matched to models by filename: data/samples/<model>.npz
pairs with models/<model>_meta.json. Labels (e.g. kws y_test) are intentionally ignored --
the demo reports predicted classes only.

    uv run python src/common/prepare_samples.py --out .cluster/shared/samples
"""
import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))


def pick_inputs(npz):
    """The sample tensor is the largest uint8 array with ndim >= 2 (skips label vectors)."""
    best = None
    for k in npz.files:
        a = npz[k]
        if a.dtype == np.uint8 and a.ndim >= 2:
            if best is None or a.size > npz[best].size:
                best = k
    if best is None:
        raise SystemExit("no uint8 (ndim>=2) sample array found; keys=%s" % list(npz.files))
    return best, npz[best]


def convert(npz_path, models_dir, out_dir):
    model = os.path.basename(npz_path)[:-4]  # strip .npz
    meta_path = os.path.join(models_dir, model + "_meta.json")
    if not os.path.isfile(meta_path):
        raise SystemExit("no metadata for %s (expected %s)" % (model, meta_path))
    meta = json.load(open(meta_path))
    shape = meta["input_shape"]
    n = int(np.prod(shape))

    with np.load(npz_path) as npz:
        key, arr = pick_inputs(npz)
        per_sample = int(np.prod(arr.shape[1:]))
        if per_sample != n:
            raise SystemExit("%s: sample has %d values, model %s expects %d %s"
                             % (os.path.basename(npz_path), per_sample, model, n, shape))
        count = int(arr.shape[0])
        flat = np.ascontiguousarray(arr, dtype=np.uint8).reshape(count, n)

    os.makedirs(out_dir, exist_ok=True)
    bin_path = os.path.join(out_dir, model + ".bin")
    with open(bin_path, "wb") as fh:
        fh.write(flat.tobytes(order="C"))
    side = {"model": model, "count": count, "per_sample_bytes": n,
            "input_shape": shape, "num_classes": meta.get("num_classes"),
            "class_names": meta.get("class_names")}
    with open(os.path.join(out_dir, model + ".samples.json"), "w") as fh:
        json.dump(side, fh)
    print("%-28s %5d samples x %6d bytes  (from key %r) -> %s"
          % (model, count, n, key, bin_path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(REPO, "data", "samples"),
                    help="dir of <model>.npz source datasets")
    ap.add_argument("--models-dir", default=os.path.join(REPO, "models"),
                    help="dir of <model>_meta.json")
    ap.add_argument("--out", default=os.path.join(REPO, ".cluster", "shared", "samples"),
                    help="output dir for <model>.bin + sidecar")
    args = ap.parse_args()

    npzs = sorted(f for f in os.listdir(args.data_dir) if f.endswith(".npz")) \
        if os.path.isdir(args.data_dir) else []
    if not npzs:
        print("no .npz in %s; nothing to prepare" % args.data_dir)
        return
    for f in npzs:
        convert(os.path.join(args.data_dir, f), args.models_dir, args.out)


if __name__ == "__main__":
    main()
