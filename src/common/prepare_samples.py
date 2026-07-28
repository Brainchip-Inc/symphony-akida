"""Convert .npz sample sets into a numpy-free form the SOAM clients can serve.

The clients run under the master's Python 3.6 soamapi binding, which has no numpy -- so they
cannot read .npz directly. This host utility (run via uv, which has numpy) pre-flattens each
dataset into:

  <set>.bin           raw concatenated uint8, C-order, count * per_sample_bytes
  <set>.samples.json  {set, model, count, per_sample_bytes, input_shape, class_names,
                       source_npz, has_ground_truth}

The client then seeks into the .bin for ready-to-send byte tensors (no parsing) and ships them
as binary.

Two ways in. By convention, data/samples/<model>.npz pairs with models/<model>_meta.json and
becomes a set named after the model -- that is how the committed random fallback set is
prepared. Or explicitly, for a test kit that lives outside the repo because it is gigabytes:

    uv run python src/common/prepare_samples.py --out .cluster/shared/samples
    uv run python src/common/prepare_samples.py --npz ~/data/voc/VOCdevkit/voc2007_test_r448.npz \\
        --model tiled_yolov2_voc --out .cluster/shared/samples

Large kits are streamed in chunks rather than loaded: the full VOC2007 test split is 2.8 GiB
of frames, and materialising it plus its byte copy would cost 6 GB of RSS.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from testkit import TestKit  # noqa: E402

CHUNK = 64


def pick_inputs(npz):
    """The sample tensor is the largest uint8 array with ndim >= 2 (skips label vectors)."""
    best = None
    for key in npz.files:
        array = npz[key]
        if array.dtype == np.uint8 and array.ndim >= 2:
            if best is None or array.size > npz[best].size:
                best = key
    if best is None:
        raise SystemExit("no uint8 (ndim>=2) sample array found; keys=%s" % list(npz.files))
    return best


def convert(npz_path, model, set_name, models_dir, out_dir):
    meta_path = os.path.join(models_dir, model + "_meta.json")
    if not os.path.isfile(meta_path):
        raise SystemExit("no metadata for %s (expected %s)" % (model, meta_path))
    meta = json.load(open(meta_path))
    # Apps whose client sends a larger image than the model input (the shard app sends a full
    # 448x448x3 frame that the services later split into 224x224x3 tiles) declare a
    # "sample_input_shape"; validate against that. Classifier models omit it and fall back to
    # the model input shape, so their prep is unchanged.
    shape = meta.get("sample_input_shape") or meta["input_shape"]
    expected = int(np.prod(shape))

    kit = _open_frames(npz_path)
    frames, ground_truth = kit
    per_sample = int(np.prod(frames.shape[1:]))
    if per_sample != expected:
        raise SystemExit("%s: sample has %d values, model %s expects %d %s"
                         % (os.path.basename(npz_path), per_sample, model, expected, shape))
    count = int(frames.shape[0])

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    bin_path = os.path.join(out_dir, set_name + ".bin")
    with open(bin_path, "wb") as handle:
        for start in range(0, count, CHUNK):
            block = np.ascontiguousarray(frames[start:start + CHUNK], dtype=np.uint8)
            handle.write(block.tobytes(order="C"))
    sidecar = {"set": set_name, "model": model, "count": count,
               "per_sample_bytes": per_sample, "input_shape": list(shape),
               "num_classes": meta.get("num_classes"), "class_names": meta.get("class_names"),
               "source_npz": os.path.abspath(npz_path), "has_ground_truth": ground_truth}
    with open(os.path.join(out_dir, set_name + ".samples.json"), "w") as handle:
        json.dump(sidecar, handle)
    print("%-28s %5d samples x %6d bytes  %s-> %s"
          % (set_name, count, per_sample, "(with ground truth) " if ground_truth else "",
             bin_path))


def _open_frames(npz_path):
    """(frames, has_ground_truth): memory mapped for a test kit, plain load otherwise."""
    try:
        kit = TestKit(npz_path)
        return kit.frames, kit.has_ground_truth
    except (KeyError, ValueError):
        with np.load(npz_path) as npz:
            return npz[pick_inputs(npz)], False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(REPO, "data", "samples"),
                    help="dir of <model>.npz source datasets")
    ap.add_argument("--models-dir", default=os.path.join(REPO, "models"),
                    help="dir of <model>_meta.json")
    ap.add_argument("--out", default=os.path.join(REPO, ".cluster", "shared", "samples"),
                    help="output dir for <set>.bin + sidecar")
    ap.add_argument("--npz", help="prepare this .npz instead of scanning --data-dir")
    ap.add_argument("--model", help="model the --npz frames feed (required with --npz)")
    ap.add_argument("--name", help="sample set name (default: the --npz basename)")
    args = ap.parse_args()

    if args.npz:
        if not args.model:
            raise SystemExit("--npz needs --model (which model's input shape it must match)")
        name = args.name or os.path.basename(args.npz)[:-4]
        convert(args.npz, args.model, name, args.models_dir, args.out)
        return

    found = sorted(f for f in os.listdir(args.data_dir) if f.endswith(".npz")) \
        if os.path.isdir(args.data_dir) else []
    if not found:
        print("no .npz in %s; nothing to prepare" % args.data_dir)
        return
    for name in found:
        model = name[:-4]
        convert(os.path.join(args.data_dir, name), model, model, args.models_dir, args.out)


if __name__ == "__main__":
    main()
