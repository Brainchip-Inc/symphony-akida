"""Which models the demos surface -- shared by both dashboards and the App B service.

For now both apps show only KWS and VWW. surface_search_classifier is NOT deleted
(it stays in models/ for later); it is simply filtered out of the UI and the per-node
service listing. Widen SHOWN_MODELS to expose more models again.

Pure stdlib on purpose (no numpy/akida): this is imported by host-side dashboards
(under uv) and by the in-container HTTP server alike.
"""
import os

# Model stems (basename without .fbz) shown in both classifier apps, in display order.
SHOWN_MODELS = ["kws_keyword_spotting_sparse", "vww_person_detect"]

# Models surfaced by the image-shard-inference app only. Kept OUT of SHOWN_MODELS on
# purpose: yolo_akidanet_voc is a detector, so it would be meaningless (an argmax "class")
# in the two classifier dashboards. Its own dashboard uses this separate allowlist.
SHARD_MODELS = ["yolo_akidanet_voc"]


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
