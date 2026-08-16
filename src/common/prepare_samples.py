"""Convert .npz sample sets into a numpy-free form the SOAM clients can serve.

The clients run under the master's Python 3.6 soamapi binding, which has no numpy -- so they
cannot read .npz directly. This host utility (run via uv, which has numpy) pre-flattens each
dataset into:

  <set>.bin           raw concatenated uint8, C-order, count * per_sample_bytes
  <set>.samples.json  {set, model, count, per_sample_bytes, input_shape, class_names,
                       source_npz, has_ground_truth, random}

The client then seeks into the .bin for ready-to-send byte tensors (no parsing) and ships them
as binary. The sidecars are also what the dashboards list as selectable sample sets.

`data/` is one folder per dataset, each holding exactly one .npz (see data/README.md). The
*folder* names the set, because folder names are unique by construction while file stems are
not, and the output directory here is flat. Nothing inside a file says which model it feeds:
the per-sample size does that, matched against models/<model>_meta.json.

    # every dataset folder: data/<dataset>/<one>.npz -> the set <dataset>
    uv run python src/common/prepare_samples.py --out .cluster/shared/samples

    # one explicit file, named after itself unless --name says otherwise
    uv run python src/common/prepare_samples.py --npz ~/data/voc/VOCdevkit/voc2007_test_r448.npz \\
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
# A dataset that is uniform noise says so inside the file, so the flag travels with the bytes
# rather than with a filename an --npz/--name run could rewrite. Everything downstream keys its
# "this is not real data" warnings off the sidecar this becomes.
RANDOM_KEY = "synthetic_random"
LFS_POINTER = b"version https://git-lfs.github.com/spec/v1"


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

    This is what lets a dataset be dropped in under any name: nothing inside the .npz says
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
    frames, ground_truth, is_random = _open_frames(npz_path)
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
               "source_npz": os.path.abspath(npz_path), "has_ground_truth": ground_truth,
               "random": is_random}
    with open(os.path.join(out_dir, set_name + ".samples.json"), "w") as handle:
        json.dump(sidecar, handle)
    print("%-28s %5d samples x %6d bytes  %s-> %s"
          % (set_name, count, per_sample,
             "(RANDOM noise) " if is_random else
             ("(with ground truth) " if ground_truth else ""), bin_path))


def _is_lfs_pointer(path):
    """True for the ~130 bytes of text git-lfs leaves in place of a file it has not fetched.

    Worth its own check because both zipfile and np.load describe a pointer as a corrupt
    archive, which sends the reader hunting for the wrong problem entirely. Exact and free: the
    spec fixes the first line, and a real .npz is a zip starting "PK", so the size gate alone
    rules out every real dataset before anything is read.
    """
    if os.path.getsize(path) > 1024:
        return False
    with open(path, "rb") as handle:
        return handle.read(len(LFS_POINTER)) == LFS_POINTER


def _open_frames(npz_path):
    """(frames, has_ground_truth, is_random): memory mapped for a test kit, plain load otherwise."""
    # np.load on a non-archive tries to read it as a pickle and reports *that* instead, which is
    # a confusing thing to print at someone who half-copied a file in -- or, far likelier, at
    # someone who cloned without git-lfs and is holding pointer text.
    if not os.path.isfile(npz_path):
        raise PrepareError("no such file")
    if _is_lfs_pointer(npz_path):
        raise PrepareError("this is a Git LFS pointer, not the dataset itself; "
                           "run 'git lfs install && git lfs pull'")
    if not zipfile.is_zipfile(npz_path):
        raise PrepareError("not an .npz archive")
    try:
        kit = TestKit(npz_path)
        return kit.frames, kit.has_ground_truth, False
    except (KeyError, ValueError):
        with np.load(npz_path) as npz:
            return npz[pick_inputs(npz)], False, RANDOM_KEY in npz.files


def scan_data(data_dir, models_dir, out_dir):
    """Prepare every dataset folder: data/<dataset>/<one>.npz -> the set <dataset>.

    A launcher calls this, so one unreadable, half-copied, unfetched or wrongly-shaped dataset
    must not take the cluster down with it: say what was skipped and why, and carry on.

    A folder holding two .npz files is skipped rather than guessed at. One folder is one set,
    and there is no honest way to name two.
    """
    if not os.path.isdir(data_dir):
        print("no %s; nothing to prepare" % data_dir)
        return
    loose = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    for path in loose:
        print("%-28s skipped: datasets live in %s/<dataset>/, not loose in %s/"
              % (os.path.basename(path), os.path.basename(data_dir),
                 os.path.basename(data_dir)))
    names = sorted(name for name in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, name)))
    if not names:
        print("no dataset folders in %s; nothing to prepare" % data_dir)
        return
    for name in names:
        found = sorted(glob.glob(os.path.join(data_dir, name, "*.npz")))
        if not found:
            print("%-28s skipped: no .npz in %s/" % (name, name))
            continue
        if len(found) > 1:
            print("%-28s skipped: %d .npz files; a dataset folder holds exactly one"
                  % (name, len(found)))
            continue
        try:
            convert(found[0], None, name, models_dir, out_dir)
        except Exception as exc:
            print("%-28s skipped: %s" % (name, exc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(REPO, "data"),
                    help="dir of dataset folders (<dataset>/<one>.npz)")
    ap.add_argument("--models-dir", default=os.path.join(REPO, "models"),
                    help="dir of <model>_meta.json")
    ap.add_argument("--out", default=os.path.join(REPO, ".cluster", "shared", "samples"),
                    help="output dir for <set>.bin + sidecar")
    ap.add_argument("--npz", help="prepare this one .npz instead of scanning --data-dir")
    ap.add_argument("--model", help="model the frames feed (default: matched on input shape)")
    ap.add_argument("--name", help="sample set name (default: the --npz basename)")
    args = ap.parse_args()

    try:
        if args.npz:
            convert(args.npz, args.model, args.name or os.path.basename(args.npz)[:-4],
                    args.models_dir, args.out)
        else:
            scan_data(args.data_dir, args.models_dir, args.out)
    except PrepareError as exc:
        raise SystemExit(str(exc))


if __name__ == "__main__":
    main()
