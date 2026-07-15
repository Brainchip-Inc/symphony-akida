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
maps on the AKD1000 (so the ON-CHIP badge is truthful), only KWS + VWW are shown, and the
workload is fed from the real .npz samples (via prepare_samples.py's <model>.bin) instead
of the old fat *.samples.json int-lists.

Run (from the repo, deps already installed via uv) -- run_dashboard.sh auto-sizes the node
count (AKD1500 + AKD1000 chips, capped at 7); or set it explicitly:
    AKIDA_NODE_COUNT=$(ls -d /dev/akd1500_* /dev/akida[0-9]* 2>/dev/null | wc -l) \
        uv run python src/apps/serial-http-round-robin/dashboard/app.py   # http://localhost:5001
"""
import json
import os
import sys
import time

from flask import Flask, jsonify, render_template_string, request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "client"))     # akida_client (HTTP client)
sys.path.insert(0, os.path.join(REPO, "src", "common"))    # shared KWS+VWW allowlist
from akida_client import AkidaServiceClient, AkidaServiceError  # noqa: E402
import models as allowlist  # noqa: E402

# Prepared samples (<model>.bin + <model>.samples.json) that prepare_samples.py writes
# from the .npz into the repo-local shared dir. Same artefacts the SOAM client consumes.
SAMPLES_DIR = os.environ.get("AKIDA_SAMPLES_DIR",
                             os.path.join(REPO, ".cluster", "shared", "samples"))
SHARED_MODELS = os.environ.get("AKIDA_SHARED_MODELS",
                               os.path.join(REPO, ".cluster", "shared", "models"))
# Cap the serial workload so the round-robin demo completes promptly (the .npz sets hold
# thousands of samples). Raise AKIDA_SAMPLE_LIMIT to run more.
LIMIT = int(os.environ.get("AKIDA_SAMPLE_LIMIT", "200"))


def _discover_nodes():
    """Per-node URL list, sized to the chip count (matches up.sh's 8790+j publishing).

    Resolution order: AKIDA_NODES (explicit CSV) -> AKIDA_NODE_COUNT (one URL per chip on
    AKIDA_PORT_BASE+i, default base 8790) -> fallback three nodes on 8790-8792.
    """
    explicit = os.environ.get("AKIDA_NODES")
    if explicit is not None:
        return [u.strip() for u in explicit.split(",") if u.strip()]
    count = os.environ.get("AKIDA_NODE_COUNT")
    base = int(os.environ.get("AKIDA_PORT_BASE", "8790"))
    if count:
        return ["http://localhost:%d" % (base + i) for i in range(int(count))]
    return ["http://localhost:%d" % (base + i) for i in range(3)]


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


# ---- API ----
@app.get("/api/fleet")
def api_fleet():
    fleet = []
    for c in clients():
        try:
            fleet.append({"url": c.base_url, "up": True, **c.health()})
        except AkidaServiceError as e:
            fleet.append({"url": c.base_url, "up": False, "error": str(e)})
    return jsonify({"nodes": fleet})


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
                         "device": r.get("device")})
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
    return render_template_string(PAGE, nodes=NODES, shared=SHARED_MODELS)


PAGE = r"""<!doctype html><html><head><meta charset=utf-8>
<title>SymAkida — Neuromorphic Model Service</title>
<style>
 :root{--bg:#0b0e14;--panel:#141925;--line:#222b3d;--txt:#dbe3f0;--mut:#7b8aa6;--acc:#4f9cff;--ok:#36d399;--warn:#fbbd23;--err:#f87272}
 *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--txt);font:14px/1.5 ui-monospace,Menlo,Consolas,monospace}
 header{padding:16px 22px;border-bottom:1px solid var(--line);display:flex;align-items:baseline;gap:14px}
 header h1{font-size:18px;margin:0;letter-spacing:.5px} header .sub{color:var(--mut)}
 main{max-width:1100px;margin:0 auto;padding:22px;display:grid;gap:22px}
 .panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px}
 .panel h2{margin:0 0 12px;font-size:13px;text-transform:uppercase;letter-spacing:1px;color:var(--mut)}
 .fleet{display:flex;gap:12px;flex-wrap:wrap}
 .node{border:1px solid var(--line);border-radius:8px;padding:10px 12px;min-width:200px;background:#0f1421}
 .dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px}
 .up{background:var(--ok)} .down{background:var(--err)}
 table{width:100%;border-collapse:collapse} th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line)}
 th{color:var(--mut);font-weight:600;font-size:12px}
 button{background:#1c2333;color:var(--txt);border:1px solid var(--line);border-radius:6px;padding:5px 11px;cursor:pointer;font:inherit}
 button:hover{border-color:var(--acc)} button.acc{background:var(--acc);color:#04121f;border-color:var(--acc);font-weight:700}
 button.danger:hover{border-color:var(--err)}
 .cur{color:var(--ok)} .pill{font-size:11px;color:var(--mut);border:1px solid var(--line);border-radius:20px;padding:1px 8px}
 .hw{color:#04121f;background:var(--ok);border-radius:20px;padding:1px 9px;font-size:11px;font-weight:700}
 .sw{color:#04121f;background:var(--warn);border-radius:20px;padding:1px 9px;font-size:11px;font-weight:700}
 .ready{color:var(--ok);border:1px solid var(--ok);border-radius:20px;padding:1px 9px;font-size:11px}
 input[type=text]{background:#0f1421;border:1px solid var(--line);color:var(--txt);border-radius:6px;padding:6px 9px;width:420px;font:inherit}
 .row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
 .bar{height:14px;background:var(--acc);border-radius:3px}
 .muted{color:var(--mut)} .mono{font-variant-numeric:tabular-nums}
 #log{white-space:pre-wrap;color:var(--mut);max-height:120px;overflow:auto;font-size:12px}
</style></head><body>
<header><h1>◈ SymAkida</h1><span class=sub>neuromorphic model service · load / unload / hot-swap on the Symphony Akida fleet</span></header>
<main>
 <div class=panel><h2>Fleet</h2><div class=fleet id=fleet>…</div></div>
 <div class=panel><h2>Models <span class=muted id=shared></span></h2>
   <table id=models><thead><tr><th>model</th><th>input</th><th>classes</th><th>size</th><th></th></tr></thead><tbody></tbody></table>
   <div class=row style=margin-top:12px>
     <button class=danger onclick=unload()>Unload all</button>
     <span class=muted>·</span>
     <input type=text id=stagepath placeholder="/path/to/local/model.fbz">
     <button onclick=stage()>Stage local .fbz</button>
   </div>
 </div>
 <div class=panel><h2>Sample workload across the chips</h2>
   <div class=row><select id=ds style="background:#0f1421;color:var(--txt);border:1px solid var(--line);border-radius:6px;padding:6px"></select>
     <button class=acc onclick=runsamples()>▶ Run across fleet</button>
     <span id=runsum class=muted></span></div>
   <div id=hist style=margin:14px:0></div>
   <table id=results style=margin-top:10px><thead><tr><th>#</th><th>node</th><th>class</th><th>latency µs</th><th>ran on</th></tr></thead><tbody></tbody></table>
 </div>
 <div class=panel><h2>Log</h2><div id=log></div></div>
</main>
<script>
const $=s=>document.querySelector(s), api=(p,b)=>fetch(p,b?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)}:{}).then(r=>r.json());
function log(m){$('#log').textContent=('['+new Date().toLocaleTimeString()+'] '+m+'\n')+$('#log').textContent;}
async function refresh(){
  const dsSel=$('#ds').value;
  const f=await api('/api/fleet');
  $('#fleet').innerHTML=f.nodes.map(n=>{
    let badge='';
    if(n.up){
      if(n.model){
        badge = n.akida_mapped
          ? `<span class=hw>● ON-CHIP · AKD1000</span> <span class=muted>${n.device||''}</span>`
          : `<span class=sw>▲ SOFTWARE (CPU)</span> <span class=muted title="${(n.map_error||'').replace(/"/g,'')}">not on v1 silicon</span>`;
      } else {
        badge = n.hardware_present
          ? `<span class=ready>◆ AKD1000 attached</span> <span class=muted>${n.device||''} · idle</span>`
          : `<span class=sw>no chip</span>`;
      }
    }
    return `<div class=node><span class="dot ${n.up?'up':'down'}"></span>${n.url.replace('http://','')}<br>
      <span class=muted>${n.up?(n.host||''):'down'}</span><br>
      ${n.up?('model: <b class='+(n.model?'cur':'muted')+'>'+(n.model||'—')+'</b>'):''}<br>
      ${badge}
      ${n.up&&n.akida_version?'<br><span class=pill>akida '+n.akida_version+'</span>':''}</div>`;
  }).join('');
  const m=await api('/api/models'); const cur=m.current&&m.current.name;
  $('#shared').textContent=' ';
  $('#models tbody').innerHTML=(m.models||[]).map(x=>`<tr><td>${x.name} ${x.name===cur?'<span class=pill style="color:var(--ok)">current</span>':''}</td>
    <td class=mono>${(x.input_shape||[]).join('×')||'?'}</td><td>${x.num_classes||'?'} <span class=muted>${(x.class_names||[]).slice(0,4).join(', ')}${(x.class_names||[]).length>4?'…':''}</span></td>
    <td class=mono>${x.size_bytes?(x.size_bytes/1024).toFixed(0)+'k':''}</td>
    <td><button onclick="load('${x.name}')">${x.name===cur?'Reload':'Load'}</button></td></tr>`).join('')||'<tr><td colspan=5 class=muted>no models staged</td></tr>';
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
    (nhw===r.rows.length?`<span class=hw>all ${nhw} ON-CHIP</span>`:`<span class=sw>${nhw}/${r.rows.length} on-chip</span>`);
  const mx=Math.max(1,...Object.values(r.histogram));
  $('#hist').innerHTML=Object.entries(r.histogram).map(([k,v])=>`<div class=row><span style=width:110px>${k}</span><div class=bar style=width:${v/mx*300}px></div><span class=mono>&nbsp;${v}</span></div>`).join('');
  $('#results tbody').innerHTML=r.rows.map(x=>`<tr><td class=mono>${x.i}</td><td>${(x.node||'').replace('.local','')}</td><td>${x.error?('<span style=color:var(--err)>'+x.error+'</span>'):x.cls_name}</td><td class=mono>${x.us||''}</td><td>${x.hardware?'<span class=hw>AKD1000</span>':(x.mode==='software'?'<span class=sw>CPU</span>':'')}</td></tr>`).join('');
  log('workload done: '+r.rows.length+' inferences');refresh();}
refresh();setInterval(refresh,5000);
</script></body></html>"""


if __name__ == "__main__":
    port = int(os.environ.get("FLASK_PORT", "5001"))
    print("SymAkida GUI on http://localhost:%d  (nodes: %s)" % (port, ", ".join(NODES)))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
