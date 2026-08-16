"""Convert .npz sample sets into a numpy-free form the SOAM clients can serve.

The clients run under the master's Python 3.6 soamapi binding, which has no numpy -- so they
cannot read .npz directly. This host utility (run via uv, which has numpy) pre-flattens each
dataset into:

  <set>.bin           raw concatenated uint8, C-order, count * per_sample_bytes
  <set>.samples.json  {set, model, count, per_sample_bytes, input_shape, class_names,
                       source_npz, has_ground_truth, random}

A model with no dataset of its own gets a synthesised random set instead (see synthesise),
flagged `random: true` so nothing downstream presents noise as real samples.

The client then seeks into the .bin for ready-to-send byte tensors (no parsing) and ships them
as binary. The sidecars are also what the dashboards list as selectable sample sets.

Three ways in:

    # data/samples/<model>.npz -> a set named after the model (the committed random fallback)
    uv run python src/common/prepare_samples.py --out .cluster/shared/samples

    # every kit in a directory, each named after its own file, model resolved by input shape
    uv run python src/common/prepare_samples.py --kits data/voc --out .cluster/shared/samples

    # one explicit file
    uv run python src/common/prepare_samples.py --npz ~/data/voc/voc2007_test_r448.npz \\
        --out .cluster/shared/samples

Large kits are streamed in chunks rather than loaded: the full VOC2007 test split is 2.8 GiB
of frames, and materialising it plus its byte copy would cost 6 GB of RSS.
"""
import argparse
import glob
import json
import os
import sys
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from testkit import TestKit  # noqa: E402

CHUNK = 64
META_SUFFIX = "_meta.json"
RANDOM_COUNT = 256      # enough to cycle through; the serial-http workload caps at 200 anyway


class PrepareError(Exception):
    """A dataset that cannot be prepared, carrying the reason to show the user."""


def pick_inputs(npz):
    """The sample tensor is the largest uint8 array with ndim >= 2 (skips label vectors)."""
    best = None
    for key in npz.files:
        array = npz[key]
        if array.dtype == np.uint8 and array.ndim >= 2:
            if best is None or array.size > npz[best].size:
                best = key
    if best is None:
        raise PrepareError("no uint8 (ndim>=2) sample array found; keys=%s" % list(npz.files))
    return best


def sample_shape(meta):
    """The shape one sample must have to be sent to this model.

    Apps whose client sends a larger image than the model input (the shard app sends a full
    448x448x3 frame that the services later split into 224x224x3 tiles) declare a
    "sample_input_shape"; classifier models omit it and take the model input shape as is.
    """
    return meta.get("sample_input_shape") or meta["input_shape"]


def read_meta(models_dir, model):
    path = os.path.join(models_dir, model + META_SUFFIX)
    if not os.path.isfile(path):
        raise PrepareError("no metadata for %s (expected %s)" % (model, path))
    with open(path) as handle:
        return json.load(handle)


def resolve_model(models_dir, per_sample):
    """The model a dataset feeds, matched on per-sample size.

    This is what lets a test kit be dropped in under any name: nothing inside the .npz says
    which model it belongs to, but its sample shape does.
    """
    matches = []
    for name in sorted(os.listdir(models_dir)):
        if name.endswith(META_SUFFIX):
            model = name[:-len(META_SUFFIX)]
            if int(np.prod(sample_shape(read_meta(models_dir, model)))) == per_sample:
                matches.append(model)
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise PrepareError("no model in %s takes %d values per sample" % (models_dir, per_sample))
    raise PrepareError("%d values per sample fits %s; pass --model"
                       % (per_sample, " and ".join(matches)))


def convert(npz_path, model, set_name, models_dir, out_dir):
    frames, ground_truth = _open_frames(npz_path)
    per_sample = int(np.prod(frames.shape[1:]))
    model = model or resolve_model(models_dir, per_sample)
    meta = read_meta(models_dir, model)
    shape = sample_shape(meta)
    if per_sample != int(np.prod(shape)):
        raise PrepareError("sample has %d values, model %s expects %d %s"
                           % (per_sample, model, int(np.prod(shape)), shape))
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
    # Kits are symlinked in, so a broken link is the likeliest failure and deserves to say so.
    # And np.load on a non-archive tries to read it as a pickle and reports *that* instead, which
    # is a confusing thing to print at someone who half-copied a file into the kit directory.
    if not os.path.isfile(npz_path):
        raise PrepareError("no such file (a broken symlink?)")
    if not zipfile.is_zipfile(npz_path):
        raise PrepareError("not an .npz archive")
    try:
        kit = TestKit(npz_path)
        return kit.frames, kit.has_ground_truth
    except (KeyError, ValueError):
        with np.load(npz_path) as npz:
            return npz[pick_inputs(npz)], False


def scan_kits(kits_dir, models_dir, out_dir):
    """Prepare every .npz in a directory of test kits, under whatever name each one has.

    A launcher calls this, so one unreadable, half-copied or wrongly-shaped file must not take
    the cluster down with it: say what was skipped and why, and carry on. A name already
    prepared (the committed sets go first) is left alone rather than overwritten.
    """
    found = sorted(glob.glob(os.path.join(kits_dir, "*.npz")))
    if not found:
        print("no .npz in %s; nothing to prepare" % kits_dir)
        return
    for path in found:
        name = os.path.basename(path)[:-4]
        if os.path.exists(os.path.join(out_dir, name + ".samples.json")):
            print("%-28s skipped: already prepared under that name" % name)
            continue
        try:
            convert(path, None, name, models_dir, out_dir)
        except Exception as exc:
            print("%-28s skipped: %s" % (name, exc))


def synthesise(model, models_dir, out_dir, count=RANDOM_COUNT, seed=1234):
    """Write a random-noise sample set for a model that has no dataset of its own.

    Not every model ships with samples. Without a set the serial-http workload dropdown
    cannot offer that model at all, while the batch client quietly improvises its own random
    input -- so the two apps disagree about what is runnable, for no reason the user can see.
    One synthesised set settles it: both apps run the model, over the same bytes.

    `random: true` in the sidecar is the load-bearing part. It is what stops everything
    downstream presenting noise as real samples -- the batch client's input line, the
    dropdown label, the warning above the results. Uniform uint8 exercises the whole path and
    measures throughput honestly; the predicted classes are meaningless and must say so.
    """
    meta = read_meta(models_dir, model)
    shape = sample_shape(meta)
    per_sample = int(np.prod(shape))
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    rng = np.random.default_rng(seed)          # seeded: two runs get the same noise
    bin_path = os.path.join(out_dir, model + ".bin")
    with open(bin_path, "wb") as handle:
        for start in range(0, count, CHUNK):
            block = rng.integers(0, 256, size=(min(CHUNK, count - start), per_sample),
                                 dtype=np.uint8)
            handle.write(block.tobytes(order="C"))
    sidecar = {"set": model, "model": model, "count": count,
               "per_sample_bytes": per_sample, "input_shape": list(shape),
               "num_classes": meta.get("num_classes"), "class_names": meta.get("class_names"),
               "source_npz": None, "has_ground_truth": False, "random": True}
    with open(os.path.join(out_dir, model + ".samples.json"), "w") as handle:
        json.dump(sidecar, handle)
    print("%-28s %5d samples x %6d bytes  (RANDOM noise, no dataset) -> %s"
          % (model, count, per_sample, bin_path))


def scan_missing(models_dir, out_dir, count=RANDOM_COUNT):
    """Give every model still without a set a random one, so no app has to improvise.

    Runs last, so a real dataset always wins: a model prepared by scan_models is skipped here.
    Drop a data/samples/<model>.npz in later and it takes over on the next launch.
    """
    if not os.path.isdir(models_dir):
        return
    for name in sorted(os.listdir(models_dir)):
        if not name.endswith(META_SUFFIX):
            continue
        model = name[:-len(META_SUFFIX)]
        if os.path.exists(os.path.join(out_dir, model + ".samples.json")):
            continue
        try:
            synthesise(model, models_dir, out_dir, count)
        except Exception as exc:
            print("%-28s skipped: %s" % (model, exc))


def scan_models(data_dir, models_dir, out_dir):
    """Prepare data/samples/<model>.npz, the committed sets named after their model."""
    found = sorted(f for f in os.listdir(data_dir) if f.endswith(".npz")) \
        if os.path.isdir(data_dir) else []
    if not found:
        print("no .npz in %s; nothing to prepare" % data_dir)
        return
    for name in found:
        model = name[:-4]
        convert(os.path.join(data_dir, name), model, model, models_dir, out_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(REPO, "data", "samples"),
                    help="dir of <model>.npz source datasets")
    ap.add_argument("--models-dir", default=os.path.join(REPO, "models"),
                    help="dir of <model>_meta.json")
    ap.add_argument("--out", default=os.path.join(REPO, ".cluster", "shared", "samples"),
                    help="output dir for <set>.bin + sidecar")
    ap.add_argument("--kits", help="prepare every .npz in this dir, each named after its file")
    ap.add_argument("--npz", help="prepare this .npz instead of scanning --data-dir")
    ap.add_argument("--model", help="model the frames feed (default: matched on input shape)")
    ap.add_argument("--name", help="sample set name (default: the --npz basename)")
    ap.add_argument("--random-count", type=int, default=RANDOM_COUNT,
                    help="samples in a synthesised random set (default %d)" % RANDOM_COUNT)
    ap.add_argument("--no-random", action="store_true",
                    help="do not synthesise random sets for models that have no dataset")
    args = ap.parse_args()

    try:
        if args.kits:
            scan_kits(args.kits, args.models_dir, args.out)
        elif args.npz:
            convert(args.npz, args.model, args.name or os.path.basename(args.npz)[:-4],
                    args.models_dir, args.out)
        else:
            scan_models(args.data_dir, args.models_dir, args.out)
            # Last, so a real dataset always wins the name.
            if not args.no_random:
                scan_missing(args.models_dir, args.out, args.random_count)
    except PrepareError as exc:
        raise SystemExit(str(exc))


if __name__ == "__main__":
    main()
