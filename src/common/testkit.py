"""Reader for the voc2007_test_r448 test kit .npz.

One self-contained file carries the 448 frames, the ground truth, the model configuration and
the reference detections of the tiled model, so the app can be validated and scored without
tfds, a VOC download or any akida_models install.

Two things here are worth more than they look:

* **Frames are memory mapped, not loaded.** `np.load` on an .npz is lazy per key, but each
  access re-reads the whole array -- `npz["frames"][i]` in a loop reads 3 GB per iteration.
  The members are stored uncompressed, so the frames can be mapped straight off disk instead,
  which turns a 300 ms per frame read into a free one and keeps the full 4,952 frame kit off
  the heap.
* **Ground truth is ragged**, packed as one array plus per-frame offsets, in *raw source
  pixels*. Detections must be scaled by each frame's own `raw_shapes` entry to match, never by
  448: the frames were made by stretching each source image to a square, so the inverse is
  anisotropic.
"""
import struct
import zipfile

import numpy as np


class TestKit:
    """A voc2007_test_r448 .npz, with the frames mapped rather than read."""

    def __init__(self, path):
        self.path = path
        self._npz = np.load(path)
        self.frames = _map_member(path, "frames.npy")
        self.count = len(self.frames)
        self.labels = [str(x) for x in self._npz["labels"]]
        self.raw_shapes = self._npz["raw_shapes"]
        self.has_ground_truth = "ann_offsets" in self._npz.files
        self.has_reference = "ref_offsets" in self._npz.files

    def __getitem__(self, key):
        return self._npz[key]

    @property
    def files(self):
        return self._npz.files

    def targets(self):
        """The published mAP targets, or None for a subset that does not carry them."""
        values = {name: float(self._npz[name + "_target"])
                  for name in ("map50", "map75", "map")}
        return None if any(v < 0 for v in values.values()) else values

    def annotations(self, index):
        """(boxes (M,4) xyxy in raw source pixels, labels (M,)) for one frame."""
        return self._slice(index, "ann")[:2]

    def reference(self, index):
        """(boxes (N,4) frame-normalised, scores, labels, truncated) for one frame."""
        boxes, labels, offsets = self._slice(index, "ref")
        lo, hi = offsets
        return boxes, self._npz["ref_scores"][lo:hi], labels, self._npz["ref_truncated"][lo:hi]

    def _slice(self, index, prefix):
        offsets = self._npz[prefix + "_offsets"]
        lo, hi = int(offsets[index]), int(offsets[index + 1])
        return (self._npz[prefix + "_boxes"][lo:hi], self._npz[prefix + "_labels"][lo:hi],
                (lo, hi))


_HEADER_READERS = {(1, 0): np.lib.format.read_array_header_1_0,
                   (2, 0): np.lib.format.read_array_header_2_0}


def _map_member(path, member):
    """Memory maps an uncompressed .npy member of an .npz.

    Walks the zip local file header by hand to find where the member's bytes start, then reads
    the .npy header from there; both are fixed formats, and going through ZipExtFile instead
    would only give an offset relative to the member.
    """
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo(member)
    if info.compress_type != zipfile.ZIP_STORED:
        raise ValueError("%s in %s is compressed; cannot memory map it." % (member, path))
    with open(path, "rb") as handle:
        handle.seek(info.header_offset)
        name_len, extra_len = struct.unpack("<HH", handle.read(30)[26:30])
        handle.seek(info.header_offset + 30 + name_len + extra_len)
        version = np.lib.format.read_magic(handle)
        if version not in _HEADER_READERS:
            raise ValueError("Unsupported .npy version %s in %s." % (version, member))
        shape, fortran, dtype = _HEADER_READERS[version](handle)
        offset = handle.tell()
    return np.memmap(path, dtype=dtype, shape=shape, order="F" if fortran else "C",
                     mode="r", offset=offset)


def scale_to_raw(boxes, raw_shape):
    """Frame-normalised xyxy -> raw source pixels, an anisotropic stretch on each axis."""
    height, width = int(raw_shape[0]), int(raw_shape[1])
    return np.asarray(boxes, dtype=np.float64).reshape(-1, 4) * [width, height, width, height]
