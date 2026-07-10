"""Which models the demos surface -- shared by both dashboards and the App B service.

For now both apps show only KWS and VWW. surface_search_classifier is NOT deleted
(it stays in models/ for later); it is simply filtered out of the UI and the per-node
service listing. Widen SHOWN_MODELS to expose more models again.

Pure stdlib on purpose (no numpy/akida): this is imported by host-side dashboards
(under uv) and by the in-container HTTP server alike.
"""
import os

# Model stems (basename without .fbz) shown in both apps, in display order.
SHOWN_MODELS = ["kws_keyword_spotting_sparse", "vww_person_detect"]


def _stem(name):
    name = os.path.basename(name)
    return name[:-4] if name.endswith(".fbz") else name


def is_shown(name):
    """True if the given model name/path is on the allowlist."""
    return _stem(name) in SHOWN_MODELS


def visible(models_dir):
    """Allowlisted model stems that actually have a .fbz in models_dir, in order."""
    present = set()
    if os.path.isdir(models_dir):
        present = {f[:-4] for f in os.listdir(models_dir) if f.endswith(".fbz")}
    return [m for m in SHOWN_MODELS if m in present]
