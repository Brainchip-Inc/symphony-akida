"""Laptop dashboard for the Symphony + Akida image-shard-inference app.

Same thin-viewer design as the batch-inference dashboard: it does NOT talk to Symphony itself.
It triggers the in-master shard client (`docker exec`) and renders what the client reports --
per-chip segment fan-out and the fleet throughput -- so the work (split -> 5-way on-chip
inference -> stitch) is scheduled by Symphony across every Akida chip.

    uv run python src/apps/image-shard-inference/dashboard/app.py   # http://localhost:5001
"""
import json
import os
import subprocess
import sys

from flask import Flask, jsonify, request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src", "common"))
from models import shard_visible  # noqa: E402  shard-app allowlist (yolo_akidanet_voc)

MODELS_DIR = os.environ.get("AKIDA_MODELS_DIR", os.path.join(REPO, "models"))
MASTER = os.environ.get("MASTER_CONTAINER", "symphony-master")
CLIENT = "/opt/akida-shard-client/run_client.sh"
PORT = int(os.environ.get("FLASK_PORT", "5001"))

app = Flask(__name__)


def list_models():
    return shard_visible(MODELS_DIR)


@app.route("/api/models")
def api_models():
    return jsonify({"models": list_models()})


@app.route("/api/run", methods=["POST"])
def api_run():
    body = request.get_json(force=True)
    model = body.get("model")
    count = int(body.get("count", 200))
    if model not in list_models():
        return jsonify({"error": "unknown model: %s" % model}), 400
    cmd = ["docker", "exec", MASTER, CLIENT,
           "--model", model, "--count", str(count), "--json"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired:
        return jsonify({"error": "client timed out"}), 504
    if out.returncode != 0:
        return jsonify({"error": "client failed", "detail": (out.stderr or "")[-800:]}), 500
    lines = [l for l in out.stdout.strip().splitlines() if l.strip()]
    try:
        return jsonify(json.loads(lines[-1]))
    except Exception:
        return jsonify({"error": "could not parse client output",
                        "detail": (out.stdout + out.stderr)[-800:]}), 500


@app.route("/")
def index():
    return HTML


HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Symphony + Akida shard inference</title>
<style>
  :root { --bg:#0f1420; --card:#1a2130; --fg:#e8edf5; --muted:#8b97ad; --accent:#4fd1c5; --bar:#3b82f6; }
  * { box-sizing:border-box; } body { margin:0; font:15px/1.5 system-ui,sans-serif; background:var(--bg); color:var(--fg); }
  header { padding:20px 28px; border-bottom:1px solid #26304a; }
  header h1 { margin:0; font-size:20px; } header p { margin:4px 0 0; color:var(--muted); font-size:13px; }
  main { max-width:900px; margin:0 auto; padding:24px 28px; }
  .controls { display:flex; gap:12px; align-items:flex-end; flex-wrap:wrap; background:var(--card); padding:18px; border-radius:12px; }
  label { display:block; font-size:12px; color:var(--muted); margin-bottom:5px; }
  select,input { background:#0f1420; color:var(--fg); border:1px solid #2b3550; border-radius:8px; padding:9px 11px; font-size:14px; }
  button { background:var(--accent); color:#04201d; border:0; border-radius:8px; padding:10px 20px; font-weight:600; cursor:pointer; font-size:14px; }
  button:disabled { opacity:.5; cursor:default; }
  .stats { display:flex; gap:14px; margin:20px 0; flex-wrap:wrap; }
  .stat { background:var(--card); border-radius:12px; padding:16px 20px; flex:1; min-width:150px; }
  .stat .n { font-size:28px; font-weight:700; } .stat .l { color:var(--muted); font-size:12px; }
  .stat .n small { font-size:14px; color:var(--muted); font-weight:400; }
  .card { background:var(--card); border-radius:12px; padding:18px 20px; margin-bottom:16px; }
  .card h2 { margin:0 0 14px; font-size:14px; color:var(--muted); font-weight:600; text-transform:uppercase; letter-spacing:.04em; }
  .row { display:flex; align-items:center; gap:10px; margin:6px 0; font-size:13px; }
  .row .name { width:150px; color:var(--muted); } .row .track { flex:1; background:#0f1420; border-radius:5px; overflow:hidden; height:20px; }
  .row .fill { background:var(--bar); height:100%; border-radius:5px; }
  .row .val { width:150px; text-align:right; font-variant-numeric:tabular-nums; }
  #msg { color:var(--muted); font-size:13px; margin:14px 0; }
  .err { color:#f87171; }
</style></head><body>
<header>
  <h1>Symphony + Akida &mdash; image-shard inference</h1>
  <p>Each 448&times;448 image is split into five 224&times;224 segments; Symphony fans the segments across the Akida
     chips (one on-chip inference each) and a stitch service merges the five outputs into one result.</p>
</header>
<main>
  <div class="controls">
    <div><label>Model</label><select id="model"></select></div>
    <div><label>Images</label><input id="count" type="number" value="200" min="1" step="50" style="width:120px"></div>
    <button id="run">Run shard pipeline</button>
  </div>
  <div id="msg"></div>
  <div id="out" style="display:none">
    <div class="stats">
      <div class="stat"><div class="n" id="s_thru"></div><div class="l">images / sec</div></div>
      <div class="stat"><div class="n" id="s_chips"></div><div class="l">Akida chips used</div></div>
      <div class="stat"><div class="n" id="s_speed"></div><div class="l">vs a single chip</div></div>
      <div class="stat"><div class="n" id="s_wall"></div><div class="l">wall time</div></div>
    </div>
    <div class="card"><h2>Per-chip segment distribution</h2><div id="hosts"></div></div>
    <div class="card"><h2>Detected classes (stitched, random input)</h2><div id="classes"></div></div>
  </div>
</main>
<script>
async function loadModels() {
  const r = await fetch('/api/models'); const d = await r.json();
  const sel = document.getElementById('model');
  sel.innerHTML = d.models.map(m => `<option>${m}</option>`).join('');
}
function bars(el, entries, unit) {
  const max = Math.max(1, ...entries.map(e => e[1]));
  el.innerHTML = entries.map(([name, val, extra]) =>
    `<div class="row"><div class="name">${name}</div>
     <div class="track"><div class="fill" style="width:${100*val/max}%"></div></div>
     <div class="val">${val.toLocaleString()}${unit||''}${extra?(' &middot; '+extra):''}</div></div>`).join('');
}
async function run() {
  const btn = document.getElementById('run'), msg = document.getElementById('msg');
  const model = document.getElementById('model').value, count = document.getElementById('count').value;
  btn.disabled = true; document.getElementById('out').style.display='none';
  msg.className=''; msg.textContent = `Running ${count} images through the shard pipeline (${model})…`;
  try {
    const r = await fetch('/api/run', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({model, count:+count})});
    const d = await r.json();
    if (d.error) { msg.className='err'; msg.textContent = d.error + (d.detail? (' — '+d.detail):''); btn.disabled=false; return; }
    msg.textContent = `${d.images_done} images (${d.segments_done} segments) on ${d.chips} chips · avg ${d.avg_boxes} boxes/image · ${d.image_errors} errors.`;
    document.getElementById('s_thru').innerHTML = d.throughput.toLocaleString();
    document.getElementById('s_chips').textContent = d.chips;
    document.getElementById('s_speed').innerHTML = d.speedup + '<small>×</small>';
    document.getElementById('s_wall').innerHTML = d.wall_s + '<small>s</small>';
    const hosts = Object.entries(d.per_host).sort((a,b)=>a[0].localeCompare(b[0]))
      .map(([h,v]) => [h.replace('.local',''), v.tasks, v.avg_ms+' ms']);
    bars(document.getElementById('hosts'), hosts, '');
    const cls = Object.entries(d.classes).sort((a,b)=>b[1]-a[1]).slice(0,12);
    bars(document.getElementById('classes'), cls, '');
    document.getElementById('out').style.display='';
  } catch(e) { msg.className='err'; msg.textContent = 'request failed: '+e; }
  btn.disabled = false;
}
document.getElementById('run').onclick = run;
loadModels();
</script>
</body></html>"""

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
