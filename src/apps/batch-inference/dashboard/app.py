"""Laptop dashboard for the Symphony + Akida batch-inference app.

A thin viewer: it does NOT talk to Symphony itself. It triggers the in-master
SOAM client (`docker exec`) and renders the per-chip fan-out the client reports,
so the work is scheduled by Symphony across every Akida chip -- not round-robined
by this dashboard (which is what the old demo did).

The Compute-nodes card is the exception, and the reason the page is not blank before a
run: src/common/fleet.py answers it host-side from `docker inspect` plus the service
instance logs, so the fleet, its chips and their devices are on screen the moment you open
the page. Same card, same wording, in all three app dashboards (src/common/dashboard_ui.py).

    uv run python src/apps/batch-inference/dashboard/app.py   # http://localhost:5001
"""
import json
import os
import subprocess
import sys

from flask import Flask, jsonify, request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src", "common"))
from models import visible  # noqa: E402  shared classifier-model allowlist
import dashboard_ui as ui  # noqa: E402  shared theme + Compute-nodes card
import fleet  # noqa: E402  host-side node discovery

APP = "batch-inference"
MODELS_DIR = os.environ.get("AKIDA_MODELS_DIR", os.path.join(REPO, "models"))
SHARED = os.environ.get("AKIDA_SHARED_DIR", os.path.join(REPO, ".cluster", "shared"))
MASTER = os.environ.get("MASTER_CONTAINER", "symphony-master")
CLIENT = "/opt/akida-client/run_client.sh"
PORT = int(os.environ.get("FLASK_PORT", "5001"))

app = Flask(__name__)


def list_models():
    # Only the allowlisted models (src/common/models.py) actually present in models/.
    return visible(MODELS_DIR)


@app.route("/api/models")
def api_models():
    return jsonify({"models": list_models()})


@app.route("/api/fleet")
def api_fleet():
    # Polled every 5s, so fleet.read() never raises -- it reports trouble in `error` and the
    # card shows it. Returning 500 here would just spam the console every five seconds.
    return jsonify(fleet.read(SHARED, APP))


@app.route("/api/run", methods=["POST"])
def api_run():
    body = request.get_json(force=True)
    model = body.get("model")
    count = int(body.get("count", 500))
    if model not in list_models():
        return jsonify({"error": "unknown model: %s" % model}), 400
    cmd = ["docker", "exec", MASTER, CLIENT,
           "--model", model, "--count", str(count), "--json"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
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
    return PAGE


APP_CSS = """
  .controls { margin-bottom:16px; }
"""

APP_HTML = """
  <div class="controls">
    <div><label for="model">Model</label><select id="model"></select></div>
    <div><label for="count">Samples</label><input id="count" type="number" value="2000" min="1" step="100" style="width:120px"></div>
    <button id="run">Run batch</button>
  </div>
  <div id="msg"></div>
  <div id="out" style="display:none">
    <div class="stats">
      <div class="stat"><div class="n" id="s_thru"></div><div class="l">inferences / sec</div></div>
      <div class="stat"><div class="n" id="s_chips"></div><div class="l">Akida chips used</div></div>
      <div class="stat"><div class="n" id="s_speed"></div><div class="l">vs a single chip</div></div>
      <div class="stat"><div class="n" id="s_wall"></div><div class="l">wall time</div></div>
    </div>
    <div class="card"><h2>Per-chip task distribution</h2><div id="hosts"></div></div>
    <div class="card"><h2>Predicted classes</h2><div id="classes"></div></div>
  </div>
"""

APP_JS = """
async function loadModels() {
  const r = await fetch('/api/models'); const d = await r.json();
  document.getElementById('model').innerHTML = d.models.map(m => `<option>${m}</option>`).join('');
}
async function run() {
  const btn = document.getElementById('run'), msg = document.getElementById('msg');
  const model = document.getElementById('model').value, count = document.getElementById('count').value;
  btn.disabled = true; document.getElementById('out').style.display='none';
  msg.className=''; msg.textContent = `Submitting ${count} tasks for ${model} across the fleet…`;
  try {
    const r = await fetch('/api/run', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({model, count:+count})});
    const d = await r.json();
    if (d.error) { msg.className='err'; msg.textContent = d.error + (d.detail? (': '+d.detail):''); btn.disabled=false; return; }
    msg.textContent = `${d.done} tasks done on ${d.chips} chips (${d.errors} errors).`;
    document.getElementById('s_thru').innerHTML = Math.round(d.throughput).toLocaleString();
    document.getElementById('s_chips').textContent = d.chips;
    document.getElementById('s_speed').innerHTML = d.speedup + '<small>×</small>';
    document.getElementById('s_wall').innerHTML = d.wall_s + '<small>s</small>';
    // Same numbers onto the node cards, so "which chips ran this" is answered where the
    // fleet already is rather than only in the bar chart below.
    const work = {};
    Object.entries(d.per_host).forEach(([h,v]) =>
      work[h] = {tasks: v.tasks, avg_ms: v.avg_ms, unit: 'tasks'});
    renderFleet(null, work);
    const hosts = Object.entries(d.per_host).sort((a,b)=>a[0].localeCompare(b[0]))
      .map(([h,v]) => [h.replace('.local',''), v.tasks,
                       v.avg_ms + ' ms' + (v.product ? ' · ' + v.product : '')]);
    bars(document.getElementById('hosts'), hosts, '');
    const cls = Object.entries(d.classes).sort((a,b)=>b[1]-a[1]).slice(0,12);
    bars(document.getElementById('classes'), cls, '');
    document.getElementById('out').style.display='';
  } catch(e) { msg.className='err'; msg.textContent = 'request failed: '+e; }
  btn.disabled = false;
}
document.getElementById('run').onclick = run;
loadModels();
pollFleet(); setInterval(pollFleet, 5000);
"""

# Concatenated, never %-formatted: the CSS is full of {braces} and the JS of ${literals}
# and `width:...%`, all of which a format string would mangle. See src/common/dashboard_ui.py.
PAGE = ('<!doctype html>\n<html><head><meta charset="utf-8">'
        '<title>Symphony + Akida fleet</title>\n<style>'
        + ui.BASE_CSS + APP_CSS +
        '</style></head><body>\n'
        '<header>\n'
        '  <h1>Symphony + Akida &mdash; fleet inference</h1>\n'
        '  <p>Submit a batch as one Symphony SOAM session; the session manager fans the tasks'
        ' across every Akida chip in parallel.</p>\n'
        '</header>\n<main>\n'
        + ui.FLEET_HTML + APP_HTML +
        '</main>\n<script>\n'
        + ui.FLEET_JS + ui.BARS_JS + APP_JS +
        '\n</script>\n</body></html>')

if __name__ == "__main__":
    # threaded (the Flask default) is load-bearing: /api/run blocks this request for the whole
    # batch, and the fleet card has to keep polling while it does.
    app.run(host="0.0.0.0", port=PORT, threaded=True)
