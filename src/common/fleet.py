"""Who is in the fleet, answered host-side, for the dashboards that cannot ask the nodes.

serial-http can just call /health on each node. The batch-inference and image-shard-inference
dashboards have no channel to a compute node at all -- they shell out to `docker exec
symphony-master <client>` and render the one JSON line it prints when the run finishes -- so
until now they showed nothing whatsoever until a batch completed. This module gives them the
same Compute-nodes card from two sources that are already on the host:

  docker inspect   the roster: which compute containers exist, whether they are RUNNING, the
                   chip up.sh pinned to each (AKIDA_CHIP_NODE) and any published port. This is
                   the liveness signal -- it is the one thing that knows a node died.
  the SI log       whether that node's worker actually got its chip and mapped its model.
                   `worker READY` is only written after a strict hw_only map (the worker
                   _fail()s otherwise), so it genuinely means on-chip, and up.sh already
                   waits on exactly this marker.

Pure stdlib, host-side. Deliberately NOT a file the workers publish: a file cannot know its
writer died, so it would need a heartbeat in three workers plus a TTL -- an approximation of
what `docker inspect` answers exactly.

Everything fails soft. These endpoints are polled every 5s, so a missing docker binary or a
wedged daemon must degrade to an error string in the card, never to an exception or a 500.
"""
import glob
import json
import os
import re
import subprocess
import time

import akida_product

# docker's --filter name= is a SUBSTRING match, so an unrelated "my-symphony-compute-test"
# would join the fleet. Anchor it here instead.
CONTAINER_RE = re.compile(r"^symphony-compute-(\d+)$")
# "[akida-chip] symphony-compute-0.local -> device[0] PCIe/AKD1500/16MB/0 (AKD1500)".
# The product suffix is optional so a log from an older image still parses.
DEVICE_RE = re.compile(r"->\s*device\[\d+\]\s+(\S+)(?:\s+\(([^)]+)\))?")
MAPPED_RE = re.compile(r"\[akida-chip\] mapped (\S+) ")
# si-symphony-compute-0-322218.log / http-symphony-compute-0-7778.log -> symphony-compute-0.
# Same shape up.sh's ready_hosts() strips, and the short name `hostname -s` produced.
LOG_HOST_RE = re.compile(r"^[a-z]+-(.+)-\d+\.log$")

# app -> (log glob under the shared dir, marker meaning "this node has its chip, mapped").
LOG_SPECS = {
    "batch-inference": ("soam/akida-service/logs/si-*.log", "worker READY"),
    "image-shard-inference": ("soam/shard-inference/logs/si-*.log", "worker READY"),
    "serial-http-round-robin": ("soam/http-service/logs/http-*.log", "listening on"),
}

_INSPECT_FORMAT = ('{"name":{{json .Name}},"host":{{json .Config.Hostname}},'
                   '"running":{{json .State.Running}},"env":{{json .Config.Env}},'
                   '"ports":{{json .NetworkSettings.Ports}}}')
_HTTP_PORT = "8790/tcp"      # the serial-http per-node server, published as PORT_BASE+j
# Every line we parse -- the wrapper banner, the device line, the mapped line, the READY
# marker -- is written during worker startup, before a single task, so a small head is enough
# however long the log later grows. It also keeps scanning ALL of a host's logs cheap, which
# is required: capping the scan silently reports a healthy node as not ready (a host here has
# 26 logs and the only READY one is 21st by mtime).
_LOG_HEAD = 16384
_TTL = 2.0                   # seconds; several browser tabs at 5s must not multiply docker calls

_cache = {}


def _docker(args, timeout=5):
    """Run a docker command and return stdout. Raises on failure -- callers fail soft."""
    out = subprocess.run(["docker"] + args, capture_output=True, text=True, timeout=timeout)
    if out.returncode != 0:
        raise RuntimeError((out.stderr or out.stdout or "docker failed").strip().splitlines()[-1])
    return out.stdout


def roster():
    """The compute containers, in node order, from one `docker ps` + one `docker inspect`.

    Each entry: name, host (Config.Hostname -- byte-identical to the `host` the workers
    report and to the per_host keys both SOAM clients use, so no fuzzy joining anywhere),
    chip_node, product (ASSIGNED, from the /dev node), running, url.
    """
    ids = _docker(["ps", "-aq", "--filter", "name=symphony-compute-"]).split()
    if not ids:
        return []
    nodes = []
    for line in _docker(["inspect", "--format", _INSPECT_FORMAT] + ids).splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            c = json.loads(line)
        except ValueError:
            continue
        name = str(c.get("name") or "").lstrip("/")
        match = CONTAINER_RE.match(name)
        if not match:
            continue
        env = {}
        for item in c.get("env") or []:
            key, _, value = str(item).partition("=")
            env[key] = value
        chip_node = env.get("AKIDA_CHIP_NODE") or None
        binding = (c.get("ports") or {}).get(_HTTP_PORT) or []
        url = None
        if binding and binding[0].get("HostPort"):
            url = "http://localhost:%s" % binding[0]["HostPort"]
        nodes.append({
            "index": int(match.group(1)),
            "name": name,
            "host": c.get("host") or (name + ".local"),
            "chip_node": chip_node,
            "product": akida_product.from_chip_node(chip_node),
            "running": bool(c.get("running")),
            "url": url,
        })
    nodes.sort(key=lambda n: n["index"])
    return nodes


def _log_host(path):
    match = LOG_HOST_RE.match(os.path.basename(path))
    return match.group(1) if match else None


def _head(path):
    try:
        with open(path, "r", errors="replace") as handle:
            return handle.read(_LOG_HEAD)
    except OSError:
        return None


def workers(shared_dir, app):
    """Per short-hostname worker state, from that host's most recent log that got a chip.

    Picking the newest log outright is wrong, and spectacularly so. SOAM leaves its losers
    behind: this host has 70 logs for 6 nodes because register.sh bounces the app to
    re-place instances under EqualFreeSlot, and every later instance then dies on "no Akida
    device present" -- the ORIGINAL worker still holds the chip and is the live one. So the
    newest file per host is typically a corpse, and reading it reports a healthy fleet as
    zero nodes ready with no device.

    Rule: newest log that reached `marker`, falling back to the newest log so a node that is
    genuinely still starting is still described. Everything here belongs to the current
    launch -- up.sh wipes .cluster/ before it starts -- so there is no cross-launch staleness.
    """
    spec = LOG_SPECS.get(app)
    if not spec:
        return {}
    pattern, marker = spec
    by_host = {}
    for path in glob.glob(os.path.join(shared_dir, pattern)):
        host = _log_host(path)
        if not host:
            continue
        try:
            by_host.setdefault(host, []).append((os.path.getmtime(path), path))
        except OSError:
            continue

    out = {}
    for host, entries in by_host.items():
        entries.sort(reverse=True)                       # newest first
        chosen, ready = None, False
        for _, path in entries:
            text = _head(path)
            if text is None:
                continue
            if chosen is None:
                chosen = text                            # newest readable, the fallback
            if marker in text:
                chosen, ready = text, True
                break
        if chosen is None:
            continue
        info = {"ready": ready}
        match = DEVICE_RE.search(chosen)
        if match:
            info["device"] = match.group(1)
            info["product"] = match.group(2) or akida_product.from_desc(match.group(1))
        match = MAPPED_RE.search(chosen)
        if match:
            info["model"] = match.group(1)
        out[host] = info
    return out


def _merge(node, worker):
    """One node record in the shape src/common/dashboard_ui.py's renderFleet() consumes."""
    worker = worker or {}
    ready = bool(worker.get("ready")) and node["running"]
    record = {
        "name": node["name"],
        "host": node["host"],
        "chip_node": node["chip_node"],
        # What the chip said about itself, when a worker has opened it; otherwise the family
        # up.sh assigned. Both agree in practice (the /dev prefix IS the PCI id), but only the
        # first is the device's own claim.
        "product": worker.get("product") or node["product"],
        "device": worker.get("device"),
        "url": node["url"],
        "state": "ready" if ready else ("idle" if node["running"] else "down"),
        "lines": [],
        "badges": [],
    }
    if worker.get("model"):
        record["lines"].append("model: " + worker["model"])
    if ready:
        # `worker READY` is written only after a strict hw_only map, so this is not a guess.
        record["badges"].append({"text": "ON-CHIP", "kind": "hw"})
    return record


def summary(nodes):
    products = {}
    for node in nodes:
        name = node.get("product")
        if name:
            products[name] = products.get(name, 0) + 1
    return {"total": len(nodes),
            "ready": sum(1 for n in nodes if n.get("state") == "ready"),
            "products": products}


def read(shared_dir, app):
    """{nodes, summary, error} for the Compute-nodes card. Never raises."""
    key = (shared_dir, app)
    hit = _cache.get(key)
    now = time.time()
    if hit and now - hit[0] < _TTL:
        return hit[1]
    try:
        nodes = roster()
        error = None
    except Exception as exc:                                  # noqa: BLE001 - polled endpoint
        nodes, error = [], "docker: %s" % exc
    seen = workers(shared_dir, app) if nodes else {}
    merged = [_merge(n, seen.get(n["name"])) for n in nodes]
    result = {"nodes": merged, "summary": summary(merged), "error": error}
    _cache[key] = (now, result)
    return result
