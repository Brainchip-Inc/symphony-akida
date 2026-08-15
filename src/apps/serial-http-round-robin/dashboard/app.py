"""SymAkida control-plane GUI (serial-http-round-robin app) -- runs on the laptop.

This is the restored "before" dashboard: a small Flask control plane that talks to the
per-node Akida HTTP inference servers (one per compute chip, published on host ports
8790, 8791, ...) to:

  * show fleet + per-node ON-CHIP vs SOFTWARE status
  * list the KWS/VWW models available in the shared models dir
  * load / unload / hot-swap the model on every live node
  * stage a local .fbz into the cluster's shared models dir
  * run a sample workload fanned across the fleet -- one HTTP /infer at a time,
    ROUND-ROBIN across nodes (the deliberate contrast with the batch-inference app's
    concurrent SOAM fan-out)

It differs from the archived original in three fixed behaviours: inference now genuinely
maps on the chip (so the ON-CHIP badge is truthful), only KWS + VWW are shown, and the
workload is fed from the real .npz samples (via prepare_samples.py's <model>.bin) instead
of the old fat *.samples.json int-lists.

Alone among the three dashboards this one has a live channel to every node -- /health -- so
its Compute-nodes card is built from what each chip says about itself rather than from
docker. The card, its wording and the theme are the shared ones every app uses
(src/common/dashboard_ui.py), so the three demos read as one product.

Run it through run_dashboard.sh (from the repo, deps already installed via uv): it discovers
the nodes and their published ports from the running containers, and serves
http://localhost:5001. To point it somewhere else, or when docker is not reachable:
    AKIDA_NODES=http://host-a:8790,http://host-b:8790 uv run python .../dashboard/app.py
    AKIDA_NODE_COUNT=7 uv run python .../dashboard/app.py     # ports 8790..8796
"""
import json
import os
import sys
import time

from flask import Flask, jsonify, request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "client"))     # akida_client (HTTP client)
sys.path.insert(0, os.path.join(REPO, "src", "common"))    # shared KWS+VWW allowlist
from akida_client import AkidaServiceClient, AkidaServiceError  # noqa: E402
import models as allowlist  # noqa: E402
import dashboard_ui as ui  # noqa: E402  shared theme + Compute-nodes card
import fleet  # noqa: E402  container roster (node names + published ports)

# Prepared samples (<model>.bin + <model>.samples.json) that prepare_samples.py writes
# from the .npz into the repo-local shared dir. Same artefacts the SOAM client consumes.
SAMPLES_DIR = os.environ.get("AKIDA_SAMPLES_DIR",
                             os.path.join(REPO, ".cluster", "shared", "samples"))
SHARED_MODELS = os.environ.get("AKIDA_SHARED_MODELS",
                               os.path.join(REPO, ".cluster", "shared", "models"))
# Cap the serial workload so the round-robin demo completes promptly (the .npz sets hold
# thousands of samples). Raise AKIDA_SAMPLE_LIMIT to run more.
LIMIT = int(os.environ.get("AKIDA_SAMPLE_LIMIT", "200"))


def _roster():
    """The compute containers that publish a per-node HTTP port, or [] if docker cannot say."""
    try:
        return [n for n in fleet.roster() if n["url"]]
    except Exception:
        return []


def _discover_nodes():
    """Per-node URL list.

    Resolution order: AKIDA_NODES (explicit CSV) -> AKIDA_NODE_COUNT (one URL per chip on
    AKIDA_PORT_BASE+i) -> the running containers' actual published ports -> three nodes on
    8790-8792. The container roster is preferred over counting /dev nodes on the host, which
    is what this used to do and which gets the count wrong on a mixed AKD1500/AKD1000 host --
    and it finds a node published on a port that no PORT_BASE+j arithmetic would predict.
    """
    explicit = os.environ.get("AKIDA_NODES")
    if explicit is not None:
        return [u.strip() for u in explicit.split(",") if u.strip()]
    base = int(os.environ.get("AKIDA_PORT_BASE", "8790"))
    count = os.environ.get("AKIDA_NODE_COUNT")
    if count:
        return ["http://localhost:%d" % (base + i) for i in range(int(count))]
    urls = [n["url"] for n in _roster()]
    return urls or ["http://localhost:%d" % (base + i) for i in range(3)]


NODES = _discover_nodes()

app = Flask(__name__)


def clients():
    return [AkidaServiceClient(u, SHARED_MODELS, timeout=8) for u in NODES]


def live_clients():
    out = []
    for c in clients():
        try:
            c.health()
            out.append(c)
        except AkidaServiceError:
            pass
    return out


def _node_record(url, health, roster_by_url):
    """One node in the shape src/common/dashboard_ui.py's renderFleet() consumes.

    Built from /health, which is this app's advantage: it is the chip's own account of
    itself, live, including whether the model really mapped hw_only.
    """
    known = roster_by_url.get(url) or {}
    if health is None:
        return {"name": known.get("name") or url.replace("http://", ""),
                "host": known.get("host"), "chip_node": known.get("chip_node"),
                "product": known.get("product"), "device": None, "url": url,
                "state": "down", "lines": [url.replace("http://", "")], "badges": []}

    host = health.get("host") or ""
    mapped = bool(health.get("akida_mapped"))
    lines = ["model: " + (health.get("model") or "—")]
    if health.get("akida_version"):
        lines.append("akida " + health["akida_version"])
    if mapped:
        badges = [{"text": "ON-CHIP", "kind": "hw"}]
    elif health.get("model"):
        badges = [{"text": "SOFTWARE (CPU)", "kind": "sw"}]
    elif health.get("hardware_present"):
        badges = [{"text": "chip attached", "kind": "on"}]
    else:
        badges = [{"text": "no chip", "kind": "sw"}]
    return {
        "name": host.replace(".local", "") or known.get("name") or url.replace("http://", ""),
        "host": host or known.get("host"),
        "chip_node": health.get("chip_node") or known.get("chip_node"),
        "product": health.get("product") or known.get("product"),
        "device": health.get("device"),
        "url": url,
        "state": "ready" if mapped else "idle",
        "lines": lines,
        "badges": badges,
    }


# ---- API ----
@app.get("/api/fleet")
def api_fleet():
    roster_by_url = {n["url"]: n for n in _roster()}
    nodes = []
    for c in clients():
        try:
            health = c.health()
        except AkidaServiceError:
            health = None
        nodes.append(_node_record(c.base_url, health, roster_by_url))
    return jsonify({"nodes": nodes, "summary": fleet.summary(nodes), "error": None})


@app.get("/api/models")
def api_models():
    for c in live_clients():
        try:
            m = c.list_models()
            m["models"] = [x for x in m.get("models", []) if allowlist.is_shown(x.get("name", ""))]
            return jsonify(m)
        except AkidaServiceError:
            continue
    return jsonify({"models": [], "current": None, "error": "no live node"})


@app.get("/api/samples")
def api_samples():
    """List prepared .npz-derived datasets (KWS/VWW) with their offered sample count."""
    out = []
    if os.path.isdir(SAMPLES_DIR):
        for f in sorted(os.listdir(SAMPLES_DIR)):
            if not f.endswith(".samples.json"):
                continue
            try:
                d = json.load(open(os.path.join(SAMPLES_DIR, f)))
            except Exception:
                continue
            if not allowlist.is_shown(d.get("model", "")):
                continue
            total = int(d.get("count", 0))
            out.append({"file": f, "model": d["model"],
                        "n": min(total, LIMIT), "total": total,
                        "input_shape": d.get("input_shape")})
    return jsonify({"datasets": out})


@app.post("/api/load")
def api_load():
    name = (request.json or {}).get("name")
    results = []
    for c in live_clients():
        try:
            results.append({"url": c.base_url, "ok": True, "model": c.load(name)["model"]})
        except AkidaServiceError as e:
            results.append({"url": c.base_url, "ok": False, "error": str(e)})
    return jsonify({"results": results})


@app.post("/api/unload")
def api_unload():
    results = []
    for c in live_clients():
        try:
            c.unload(); results.append({"url": c.base_url, "ok": True})
        except AkidaServiceError as e:
            results.append({"url": c.base_url, "ok": False, "error": str(e)})
    return jsonify({"results": results})


@app.post("/api/stage")
def api_stage():
    path = (request.json or {}).get("path", "").strip()
    try:
        res = AkidaServiceClient(NODES[0], SHARED_MODELS).stage_local_fbz(path)
        return jsonify({"ok": True, **res})
    except (AkidaServiceError, IndexError) as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.post("/api/run_samples")
def api_run_samples():
    """Round-robin the .npz-derived samples across the live nodes, one /infer at a time."""
    fname = (request.json or {}).get("file")
    side_path = os.path.join(SAMPLES_DIR, fname or "")
    if not fname or not os.path.isfile(side_path):
        return jsonify({"ok": False, "error": "unknown sample set"}), 400
    side = json.load(open(side_path))
    model = side["model"]
    per = int(side["per_sample_bytes"])
    total = int(side["count"])
    class_names = side.get("class_names") or []
    bin_path = os.path.join(SAMPLES_DIR, model + ".bin")
    if not os.path.isfile(bin_path):
        return jsonify({"ok": False, "error": "no .bin for %s (run prepare_samples.py)" % model}), 400
    n = min(total, LIMIT)
    with open(bin_path, "rb") as fh:
        blob = fh.read(n * per)

    nodes = live_clients()
    if not nodes:
        return jsonify({"ok": False, "error": "no live node"}), 503
    # ensure the model is mapped on every live node first
    load_errs = []
    for c in nodes:
        try:
            c.load(model)
        except AkidaServiceError as e:
            load_errs.append({"url": c.base_url, "error": str(e)})

    rows, hist = [], {}
    t0 = time.time()
    for i in range(n):
        vals = list(blob[i * per:(i + 1) * per])   # raw bytes -> ints 0..255
        c = nodes[i % len(nodes)]                   # round-robin across the fleet
        try:
            r = c.infer(vals)
            rows.append({"i": i, "node": r["host"], "cls": r["cls"],
                         "cls_name": r["cls_name"], "us": r["inference_us"],
                         "hardware": r.get("hardware"), "mode": r.get("mode"),
                         "device": r.get("device"), "product": r.get("product")})
            hist[r["cls_name"]] = hist.get(r["cls_name"], 0) + 1
        except AkidaServiceError as e:
            rows.append({"i": i, "node": c.base_url, "error": str(e)})
    wall = time.time() - t0
    lat = [r["us"] for r in rows if "us" in r]
    return jsonify({
        "ok": True, "model": model, "class_names": class_names,
        "rows": rows, "histogram": hist,
        "nodes_used": len(nodes), "wall_s": round(wall, 3),
        "avg_us": round(sum(lat) / len(lat)) if lat else None,
        "load_errors": load_errs,
    })


# ---- UI ----
@app.get("/")
def index():
    return PAGE


APP_CSS = """
  .panelrow { display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin-top:12px; }
  .pill { font-size:11px; color:var(--muted); border:1px solid var(--line); border-radius:20px;
          padding:1px 8px; }
  .cur { color:var(--good); }
  #log { white-space:pre-wrap; color:var(--muted); max-height:120px; overflow:auto;
         font-family:var(--mono); font-size:12px; }
  #hist .row .name { width:130px; }
"""

APP_HTML = """
  <div class="card"><h2>Models<span class="aside" id="shared"></span></h2>
    <table id="models"><thead><tr><th>model</th><th>input</th><th>classes</th><th>size</th><th></th></tr></thead><tbody></tbody></table>
    <div class="panelrow">
      <button class="ghost danger" onclick="unload()">Unload all</button>
      <span class="muted">·</span>
      <input type="text" id="stagepath" placeholder="/path/to/local/model.fbz" style="width:420px">
      <button class="ghost" onclick="stage()">Stage local .fbz</button>
    </div>
  </div>
  <div class="card"><h2>Sample workload across the chips</h2>
    <div class="panelrow" style="margin-top:0">
      <select id="ds"></select>
      <button onclick="runsamples()">▶ Run across fleet</button>
      <span id="runsum" class="muted"></span>
    </div>
    <div id="hist" style="margin:14px 0"></div>
    <table id="results" style="margin-top:10px"><thead><tr><th>#</th><th>node</th><th>class</th><th>latency µs</th><th>ran on</th></tr></thead><tbody></tbody></table>
  </div>
  <div class="card"><h2>Log</h2><div id="log"></div></div>
"""

APP_JS = """
const $=s=>document.querySelector(s), api=(p,b)=>fetch(p,b?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)}:{}).then(r=>r.json());
function log(m){$('#log').textContent=('['+new Date().toLocaleTimeString()+'] '+m+'\\n')+$('#log').textContent;}
async function refresh(){
  const dsSel=$('#ds').value;
  await pollFleet();
  const m=await api('/api/models'); const cur=m.current&&m.current.name;
  $('#models tbody').innerHTML=(m.models||[]).map(x=>`<tr><td>${x.name} ${x.name===cur?'<span class=pill style="color:var(--good)">current</span>':''}</td>
    <td class=mono>${(x.input_shape||[]).join('×')||'?'}</td><td>${x.num_classes||'?'} <span class=muted>${(x.class_names||[]).slice(0,4).join(', ')}${(x.class_names||[]).length>4?'…':''}</span></td>
    <td class=mono>${x.size_bytes?(x.size_bytes/1024).toFixed(0)+'k':''}</td>
    <td><button class=ghost onclick="load('${x.name}')">${x.name===cur?'Reload':'Load'}</button></td></tr>`).join('')||'<tr><td colspan=5 class=muted>no models staged</td></tr>';
  const s=await api('/api/samples'); $('#ds').innerHTML=(s.datasets||[]).map(d=>`<option value="${d.file}">${d.model} — ${d.n} samples (${d.input_shape.join('×')})</option>`).join('');
  if(dsSel) $('#ds').value=dsSel;
}
async function load(n){log('load '+n+' on fleet…');const r=await api('/api/load',{name:n});log('load: '+r.results.map(x=>x.url.replace('http://','')+(x.ok?' ✓':' ✗ '+x.error)).join('  '));refresh();}
async function unload(){log('unload all…');await api('/api/unload',{});refresh();}
async function stage(){const p=$('#stagepath').value.trim();if(!p)return;log('stage '+p);const r=await api('/api/stage',{path:p});log(r.ok?('staged: '+r.staged.join(', ')):('stage error: '+r.error));refresh();}
async function runsamples(){const f=$('#ds').value;if(!f)return;log('run workload '+f+'…');$('#runsum').textContent='running…';
  const r=await api('/api/run_samples',{file:f});
  if(!r.ok){$('#runsum').textContent='error: '+r.error;return;}
  const nhw=r.rows.filter(x=>x.hardware).length;
  $('#runsum').innerHTML=`${r.rows.length} samples · ${r.nodes_used} node(s) · ${r.wall_s}s wall · avg ${r.avg_us}µs/infer · `+
    (nhw===r.rows.length?`<span class="badge hw">all ${nhw} ON-CHIP</span>`:`<span class="badge sw">${nhw}/${r.rows.length} on-chip</span>`);
  bars($('#hist'), Object.entries(r.histogram), '');
  // Per-node tallies onto the node cards, the same way the other two dashboards do it.
  const work={};
  r.rows.forEach(x=>{ if(!x.node||x.error) return;
    const w = work[x.node] || (work[x.node]={tasks:0,_us:0});
    w.tasks++; w._us += (x.us||0); });
  Object.values(work).forEach(w=>{ w.avg_ms = +(w._us/w.tasks/1000).toFixed(3); w.unit='samples'; });
  renderFleet(null, work);
  $('#results tbody').innerHTML=r.rows.map(x=>`<tr><td class=mono>${x.i}</td><td>${(x.node||'').replace('.local','')}</td><td>${x.error?('<span class=err>'+x.error+'</span>'):x.cls_name}</td><td class=mono>${x.us||''}</td><td>${x.hardware?'<span class="badge hw">'+x.hardware+'</span>':(x.mode==='software'?'<span class="badge sw">CPU</span>':'')}</td></tr>`).join('');
  log('workload done: '+r.rows.length+' inferences');refresh();}
refresh();setInterval(refresh,5000);
"""

# Concatenated, never %-formatted: the CSS is full of {braces} and the JS of ${literals}
# and `width:...%`, all of which a format string would mangle. See src/common/dashboard_ui.py.
PAGE = ('<!doctype html>\n<html><head><meta charset="utf-8">'
        '<title>SymAkida — Neuromorphic Model Service</title>\n<style>'
        + ui.BASE_CSS + APP_CSS +
        '</style></head><body>\n'
        '<header>\n'
        '  <h1>SymAkida &mdash; neuromorphic model service</h1>\n'
        '  <p>Load, unload and hot-swap a model on every Akida chip in the fleet, then run a'
        ' sample workload round-robin across them &mdash; one HTTP /infer at a time, the'
        ' deliberate contrast with the batch-inference app\'s concurrent SOAM fan-out.</p>\n'
        '</header>\n<main>\n'
        + ui.FLEET_HTML + APP_HTML +
        '</main>\n<script>\n'
        + ui.FLEET_JS + ui.BARS_JS + APP_JS +
        '\n</script>\n</body></html>')


if __name__ == "__main__":
    port = int(os.environ.get("FLASK_PORT", "5001"))
    print("SymAkida GUI on http://localhost:%d  (nodes: %s)" % (port, ", ".join(NODES)))
    # threaded (the Flask default) is load-bearing: /api/run_samples blocks this request for
    # the whole workload, and the fleet card has to keep polling while it does.
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False, threaded=True)
