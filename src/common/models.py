"""Which models the demos surface -- shared by both dashboards and the App B service.

Pure stdlib on purpose (no numpy/akida): this is imported by host-side dashboards
(under uv) and by the in-container HTTP server alike.
"""
import os

# Model stems (basename without .fbz) shown in both classifier apps, in display order.
#
# surface_search_classifier maps hw_only on AKD1500 (verified) so it is a real on-chip
# model-management demo: load/unload/hot-swap it across the fleet and every node reports
# ON-CHIP honestly. It ships without real samples, so data/surface_search_classifier/ holds
# uniform noise instead -- committed, seeded and marked synthetic inside the .npz itself, so
# both apps run it over the same bytes and both say plainly that the class histogram means
# nothing while the throughput means everything. Replace that folder's .npz with real samples
# of the same shape and it stops being flagged. See data/README.md.
SHOWN_MODELS = ["kws_keyword_spotting_sparse", "vww_person_detect",
                "surface_search_classifier"]

# Models surfaced by the image-shard-inference app only. Kept OUT of SHOWN_MODELS on
# purpose: tiled_yolov2_voc is a detector, so it would be meaningless (an argmax "class")
# in the two classifier dashboards. Its own dashboard uses this separate allowlist.
SHARD_MODELS = ["tiled_yolov2_voc"]


def _stem(name):
    name = os.path.basename(name)
    return name[:-4] if name.endswith(".fbz") else name


def is_shown(name):
    """True if the given model name/path is on the classifier allowlist."""
    return _stem(name) in SHOWN_MODELS


def _visible(allowlist, models_dir):
    """Allowlisted stems that actually have a .fbz in models_dir, in allowlist order."""
    present = set()
    if os.path.isdir(models_dir):
        present = {f[:-4] for f in os.listdir(models_dir) if f.endswith(".fbz")}
    return [m for m in allowlist if m in present]


def visible(models_dir):
    """Classifier-app models (SHOWN_MODELS) present in models_dir, in order."""
    return _visible(SHOWN_MODELS, models_dir)


def shard_visible(models_dir):
    """Shard-app models (SHARD_MODELS) present in models_dir, in order."""
    return _visible(SHARD_MODELS, models_dir)
