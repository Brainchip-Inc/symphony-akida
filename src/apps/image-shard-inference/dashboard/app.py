"""Laptop dashboard for the Symphony + Akida image-shard-inference app.

Same thin-viewer design as the batch-inference dashboard: it does NOT talk to Symphony itself.
It triggers the in-master shard client (`docker exec`) and renders what the client reports --
per-chip tile fan-out, fleet throughput, and, when the sample set carries ground truth, the mAP
of what the six chips actually produced plus a gallery of the frames with their boxes drawn.

Everything shown comes from the run: the client dumps its merged detections, this process
scores them with the same code scripts/eval_shard_map.py uses, and draws them on the very
frames that were sent to the chips.

The Compute-nodes card is the exception, and the reason the page is not blank before a run:
src/common/fleet.py answers it host-side from `docker inspect` plus the service instance
logs, so the six chips and their devices are on screen the moment you open the page. Same
card, same wording, in all three app dashboards (src/common/dashboard_ui.py).

    uv run python src/apps/image-shard-inference/dashboard/app.py   # http://localhost:5001
"""
import glob
import json
import os
import subprocess
import sys
import threading

import numpy as np
from flask import Flask, Response, jsonify, request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src", "common"))
from detection_map import evaluate, group_by_label, summarise  # noqa: E402
from draw_detections import render  # noqa: E402
from models import shard_visible  # noqa: E402  shard-app allowlist (tiled_yolov2_voc)
from testkit import TestKit, scale_to_raw  # noqa: E402
import dashboard_ui as ui  # noqa: E402  shared theme + Compute-nodes card
import fleet  # noqa: E402  host-side node discovery

APP = "image-shard-inference"
MODELS_DIR = os.environ.get("AKIDA_MODELS_DIR", os.path.join(REPO, "models"))
SHARED = os.environ.get("AKIDA_SHARED_DIR", os.path.join(REPO, ".cluster", "shared"))
SAMPLES_DIR = os.path.join(SHARED, "samples")
RESULTS_DIR = os.path.join(SHARED, "results")
MASTER = os.environ.get("MASTER_CONTAINER", "symphony-master")
CLIENT = "/opt/akida-shard-client/run_client.sh"
PORT = int(os.environ.get("FLASK_PORT", "5001"))
GALLERY = 9

app = Flask(__name__)
# The gallery is served frame by frame after a run, so the run's detections and the kit it came
# from are held here rather than re-read per image request.
STATE = {"records": {}, "kit": None, "frames": None, "class_names": [], "dataset": None}
STATE_LOCK = threading.Lock()


def list_models():
    return shard_visible(MODELS_DIR)


def list_datasets():
    """Prepared sample sets under /shared/samples, newest first, with what each one offers."""
    out = []
    for sidecar in sorted(glob.glob(os.path.join(SAMPLES_DIR, "*.samples.json"))):
        try:
            side = json.load(open(sidecar))
        except Exception:
            continue
        if side.get("model") not in list_models():
            continue
        source = side.get("source_npz") or ""
        out.append({"name": side.get("set") or os.path.basename(sidecar).split(".")[0],
                    "model": side["model"], "count": int(side.get("count", 0)),
                    "input_shape": list(side.get("input_shape") or []),
                    "has_ground_truth": bool(side.get("has_ground_truth")),
                    "source_npz": source,
                    "is_random": not side.get("has_ground_truth")})
    out.sort(key=lambda d: (d["is_random"], -d["count"]))
    return out


def open_frames(dataset):
    """The frames the fleet was actually sent, memory mapped from the prepared .bin.

    One source for every sample set: the .bin is what the client shipped, so the gallery draws
    boxes on exactly the pixels they came from -- and it works for a set with no .npz behind it,
    which is what keeps the random fallback set showing pictures rather than broken images.
    """
    path = os.path.join(SAMPLES_DIR, dataset["name"] + ".bin")
    if not os.path.isfile(path) or len(dataset["input_shape"]) != 3:
        return None
    return np.memmap(path, dtype=np.uint8, mode="r",
                     shape=(dataset["count"],) + tuple(dataset["input_shape"]))


def find_dataset(name):
    for dataset in list_datasets():
        if dataset["name"] == name:
            return dataset
    return None


def read_dump(path):
    records = {}
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                record = json.loads(line)
                records[int(record["sample"])] = record
    return records


def score(kit, records, max_boxes, post_thresh=0.0):
    """mAP of the run, and of the kit's own reference detections through the same code.

    post_thresh gates the reference detections exactly as the run was gated. Without that the
    two rows are not comparable: the reference carries every merged box, so a run with the demo
    gate at 0.5 would look like a regression when it is only the gate doing its job. The
    reference scores already have the truncated penalty applied, so the same threshold means
    the same thing on both sides.
    """
    num_classes = len(kit.labels)
    samples = sorted(records)

    def build(boxes_of):
        detections, annotations = [], []
        for sample in samples:
            boxes, scores, labels = boxes_of(sample)
            if len(boxes):
                boxes = scale_to_raw(boxes, kit.raw_shapes[sample])
                order = np.argsort(-np.asarray(scores))[:max_boxes]
                boxes = boxes[order]
                scores = np.asarray(scores)[order]
                labels = np.asarray(labels)[order]
            else:
                boxes = np.zeros((0, 4))
                scores, labels = np.zeros((0,)), np.zeros((0,), dtype=int)
            detections.append(group_by_label(boxes, labels, num_classes, extra=scores))
            gt_boxes, gt_labels = kit.annotations(sample)
            annotations.append(group_by_label(gt_boxes, gt_labels, num_classes))
        return detections, annotations

    def from_dump(sample):
        record = records[sample]
        return record["boxes"], record["scores"], record["labels"]

    per_threshold, per_class = evaluate(*build(from_dump), num_classes=num_classes)
    result = summarise(per_threshold)
    result["per_class"] = sorted(((name, round(float(ap), 4))
                                  for name, ap in zip(kit.labels, per_class)),
                                 key=lambda kv: -kv[1])
    if kit.has_reference:
        def from_kit(sample):
            boxes, scores, labels, _ = kit.reference(sample)
            keep = np.asarray(scores) >= post_thresh
            return np.asarray(boxes)[keep], np.asarray(scores)[keep], np.asarray(labels)[keep]
        result["reference"] = summarise(evaluate(*build(from_kit),
                                                num_classes=num_classes)[0])
    # The published figures are defined on the whole split with no post-merge gate.
    result["targets"] = (kit.targets() if len(samples) == kit.count and post_thresh <= 0
                         else None)
    result["post_thresh"] = post_thresh
    return result


@app.route("/api/datasets")
def api_datasets():
    return jsonify({"models": list_models(), "datasets": list_datasets()})


@app.route("/api/fleet")
def api_fleet():
    # Polled every 5s, so fleet.read() never raises -- it reports trouble in `error` and the
    # card shows it. Returning 500 here would just spam the console every five seconds.
    return jsonify(fleet.read(SHARED, APP))


@app.route("/api/run", methods=["POST"])
def api_run():
    body = request.get_json(force=True)
    dataset = find_dataset(body.get("dataset") or "")
    if dataset is None:
        return jsonify({"error": "unknown sample set: %s" % body.get("dataset")}), 400
    count = max(1, min(int(body.get("count", 200)), dataset["count"] or 200))
    post_thresh = body.get("post_thresh")

    # --ordered so frame i is sample i, which is what lets the dump be scored against ground
    # truth and drawn on the right frames.
    cmd = ["docker", "exec", MASTER, CLIENT, "--model", dataset["model"],
           "--samples", dataset["name"], "--count", str(count), "--ordered", "--dump", "--json"]
    if post_thresh is not None:
        cmd += ["--post-thresh", str(float(post_thresh))]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    except subprocess.TimeoutExpired:
        return jsonify({"error": "client timed out"}), 504
    if out.returncode != 0:
        return jsonify({"error": "client failed", "detail": (out.stderr or "")[-800:]}), 500
    lines = [line for line in out.stdout.strip().splitlines() if line.strip()]
    try:
        result = json.loads(lines[-1])
    except Exception:
        return jsonify({"error": "could not parse client output",
                        "detail": (out.stdout + out.stderr)[-800:]}), 500

    result["dataset"] = dataset
    records = {}
    dump = result.get("dump")
    host_dump = os.path.join(RESULTS_DIR, os.path.basename(dump)) if dump else None
    if host_dump and os.path.isfile(host_dump):
        records = read_dump(host_dump)

    kit = None
    if dataset["has_ground_truth"] and os.path.isfile(dataset["source_npz"]):
        try:
            kit = TestKit(dataset["source_npz"])
        except Exception as exc:
            result["accuracy_error"] = "could not open %s: %s" % (dataset["source_npz"], exc)
    if kit is not None and records:
        try:
            result["accuracy"] = score(kit, records, int(kit["max_boxes"]),
                                       float(result.get("post_thresh") or 0.0))
        except Exception as exc:
            result["accuracy_error"] = "scoring failed: %s" % exc

    with STATE_LOCK:
        STATE["records"] = records
        STATE["kit"] = kit
        STATE["frames"] = open_frames(dataset)
        STATE["dataset"] = dataset["name"]
        STATE["class_names"] = (kit.labels if kit is not None
                                else json.load(open(os.path.join(
                                    MODELS_DIR, dataset["model"] + "_meta.json")))["class_names"])
    # Frames the gallery will show: the ones with the most detections, so the picture is not a
    # wall of empty images, but in frame order so it reads as a sample rather than a ranking.
    ranked = sorted(records, key=lambda s: -len(records[s].get("boxes") or []))[:GALLERY]
    result["gallery"] = sorted(ranked)
    result["gallery_counts"] = {str(s): len(records[s].get("boxes") or []) for s in ranked}
    result["run_token"] = os.path.basename(dump or "run")
    return jsonify(result)


@app.route("/api/frame/<int:sample>.png")
def api_frame(sample):
    with STATE_LOCK:
        kit, frames, records = STATE["kit"], STATE["frames"], STATE["records"]
        class_names = STATE["class_names"]
    if frames is None:
        return jsonify({"error": "no frames available for this sample set"}), 404
    if sample >= len(frames):
        return jsonify({"error": "sample %d out of range" % sample}), 404
    record = records.get(sample, {})
    truth = None
    if request.args.get("gt") == "1" and kit is not None and kit.has_ground_truth:
        truth = kit.annotations(sample)
    png = render(frames[sample],
                 np.asarray(record.get("boxes") or [], dtype=np.float64).reshape(-1, 4),
                 np.asarray(record.get("scores") or [], dtype=np.float64),
                 np.asarray(record.get("labels") or [], dtype=int),
                 class_names,
                 truncated=np.asarray(record.get("truncated") or [], dtype=bool),
                 truth=truth,
                 raw_shape=kit.raw_shapes[sample] if kit is not None else None, scale=1)
    return Response(png, mimetype="image/png",
                    headers={"Cache-Control": "no-store"})


@app.route("/")
def index():
    return HTML


APP_CSS = """
  .controls { margin-bottom:16px; }
  /* The accuracy table is a small figure block, not a full-width data table, so it opts out
     of the shared table rules. */
  table.map { width:auto; border-collapse:collapse; font-variant-numeric:tabular-nums; }
  table.map th, table.map td { padding:5px 16px 5px 0; text-align:right; font-size:14px;
                               border-bottom:0; }
  table.map th { color:var(--muted); font-weight:600; font-size:12px; }
  table.map td:first-child, table.map th:first-child { text-align:left; min-width:96px; }
  table.map tr.ours td { color:var(--good); font-weight:600; }
  table.map tr.tgt td { color:var(--muted); }
  .gal { display:grid; grid-template-columns:repeat(auto-fill,minmax(300px,1fr)); gap:12px; }
  .gal figure { margin:0; background:var(--bg); border-radius:10px; overflow:hidden;
                border:1px solid var(--line); }
  .gal img { display:block; width:100%; height:auto; }
  .gal figcaption { padding:7px 10px; font-size:12px; color:var(--muted);
                    display:flex; justify-content:space-between; gap:8px; }
  .toggle { display:flex; align-items:center; gap:7px; font-size:12px; color:var(--muted);
            text-transform:none; letter-spacing:0; font-weight:400; }
  .toggle input { width:14px; height:14px; accent-color:var(--accent); }
"""

APP_HTML = """
  <div class="controls">
    <div><label for="dataset">Sample set</label><select id="dataset"></select></div>
    <div><label for="count">Frames</label><input id="count" type="number" value="200" min="1" step="50" style="width:110px"></div>
    <div><label for="post">Post-merge gate</label><input id="post" type="number" value="0.5" min="0" max="1" step="0.05" style="width:110px"></div>
    <button id="run">Run shard pipeline</button>
  </div>
  <div id="banner"></div>
  <div id="msg"></div>
  <div id="out" style="display:none">
    <div class="stats" id="stats"></div>
    <div class="card" id="mapcard" style="display:none">
      <h2>Accuracy on this sample set<span class="aside" id="mapaside"></span></h2>
      <table class="map" id="maptable"></table>
    </div>
    <div class="card" id="galcard" style="display:none">
      <h2>Predictions
        <span class="aside"><label class="toggle"><input type="checkbox" id="gt"> show ground truth</label></span>
      </h2>
      <div class="gal" id="gal"></div>
      <p class="stat sub" style="margin:12px 0 0">Solid box = complete detection. Dashed = a tile
         seam still cuts it off, so its score was demoted before ranking.</p>
    </div>
    <div class="card"><h2>Per-chip tile distribution</h2><div id="hosts"></div></div>
    <div class="card" id="clscard"><h2>Detected classes</h2><div id="classes"></div></div>
  </div>
"""

APP_JS = """
let LAST = null;

async function loadDatasets() {
  const r = await fetch('/api/datasets'); const d = await r.json();
  const sel = document.getElementById('dataset');
  if (!d.datasets.length) {
    sel.innerHTML = '<option value="">no prepared sample set</option>';
    document.getElementById('run').disabled = true;
    document.getElementById('msg').textContent =
      'No sample set found under .cluster/shared/samples. Bring the cluster up first: ./scripts/launch/up.sh image-shard-inference --nodes 6';
    return;
  }
  sel.innerHTML = d.datasets.map(s =>
    `<option value="${s.name}" data-count="${s.count}" data-random="${s.is_random}"`
    + ` data-gt="${s.has_ground_truth}" data-source="${s.source_npz}">`
    + `${s.name} — ${s.count.toLocaleString()} frames${s.has_ground_truth ? ' (with ground truth)' : ''}</option>`).join('');
  sel.onchange = onDataset; onDataset();
}
function onDataset() {
  const opt = document.getElementById('dataset').selectedOptions[0];
  if (!opt) return;
  const total = +opt.dataset.count, isRandom = opt.dataset.random === 'true';
  const source = (opt.dataset.source || '').split('/').pop();
  const count = document.getElementById('count');
  count.max = total; if (+count.value > total) count.value = total;
  document.getElementById('banner').innerHTML = isRandom
    ? `<div class="note warn"><b>Random input.</b> This set is uniform 448 noise, so it contains
       no objects: an empty or near-empty result is the <i>correct</i> one, and accuracy is not
       evaluated. It exercises every stage and every throughput number. For detections and mAP,
       symlink a test kit into <code>data/voc/</code> and relaunch; every kit found there is
       offered above. See <code>data/voc/README.md</code>.</div>`
    : `<div class="note"><b>${total.toLocaleString()} real frames with ground truth</b>
       ${source ? 'from <code>'+source+'</code>' : ''}. mAP is measured on exactly the frames you
       run. The published figure is defined on the whole 4,952-frame split and with the post-merge
       gate at 0, which is how the reference measures it.</div>`;
}
function stat(n, l, sub) {
  return `<div class="stat"><div class="n">${n}</div><div class="l">${l}</div>`
       + (sub ? `<div class="sub">${sub}</div>` : '') + `</div>`;
}
function pct(x) { return (100*x).toFixed(2); }

function renderMap(a) {
  const card = document.getElementById('mapcard');
  if (!a) { card.style.display='none'; return; }
  card.style.display='';
  let rows = `<tr><th></th><th>mAP50</th><th>mAP75</th><th>mAP 50:95</th></tr>`
    + `<tr class="ours"><td>this fleet</td><td>${pct(a.map50)}</td><td>${pct(a.map75)}</td><td>${pct(a.map)}</td></tr>`;
  if (a.reference) rows += `<tr><td>reference</td><td>${pct(a.reference.map50)}</td>`
    + `<td>${pct(a.reference.map75)}</td><td>${pct(a.reference.map)}</td></tr>`;
  if (a.targets) rows += `<tr class="tgt"><td>published</td><td>${pct(a.targets.map50)}</td>`
    + `<td>${pct(a.targets.map75)}</td><td>${pct(a.targets.map)}</td></tr>`;
  else rows += `<tr class="tgt"><td colspan="4" style="text-align:left;padding-top:9px">`
    + `The published figure is defined on the whole 4,952-frame split with the post-merge gate `
    + `at 0; run it that way to compare against it.</td></tr>`;
  document.getElementById('maptable').innerHTML = rows;
  let aside = '';
  if (a.reference) {
    const d = Math.max(Math.abs(a.map50-a.reference.map50), Math.abs(a.map75-a.reference.map75),
                       Math.abs(a.map-a.reference.map));
    aside = d < 1e-9 ? 'identical to the reference detections'
                     : `differs from the reference by ${d.toExponential(1)}`;
  }
  document.getElementById('mapaside').textContent = aside;
}
function renderGallery(samples) {
  const card = document.getElementById('galcard');
  if (!samples || !samples.length) { card.style.display='none'; return; }
  card.style.display='';
  const gt = document.getElementById('gt').checked ? 1 : 0;
  const bust = (LAST && LAST.run_token) || '0';
  const counts = (LAST && LAST.gallery_counts) || {};
  document.getElementById('gal').innerHTML = samples.map(s => {
    const n = counts[s] || 0;
    return `<figure><img alt="frame ${s}" src="/api/frame/${s}.png?gt=${gt}&t=${encodeURIComponent(bust)}">
      <figcaption><span>frame ${s}</span><span>${n} ${n===1?'box':'boxes'}</span></figcaption></figure>`;
  }).join('');
}
async function run() {
  const btn = document.getElementById('run'), msg = document.getElementById('msg');
  const dataset = document.getElementById('dataset').value;
  const count = +document.getElementById('count').value;
  const post = +document.getElementById('post').value;
  btn.disabled = true; document.getElementById('out').style.display='none';
  msg.className=''; msg.textContent =
    `Running ${count.toLocaleString()} frames (${count*6} tiles) through the pipeline on ${dataset}…`;
  try {
    const r = await fetch('/api/run', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({dataset, count, post_thresh: post})});
    const d = await r.json();
    if (d.error) { msg.className='err'; msg.textContent = d.error + (d.detail? (' — '+d.detail):''); btn.disabled=false; return; }
    LAST = d;
    msg.innerHTML = `${d.images_done.toLocaleString()} frames (${d.segments_done.toLocaleString()} tiles) on `
      + `${d.chips} chips · avg ${d.avg_boxes} boxes/frame · ${d.image_errors} frame errors`
      + `${d.stalled ? ' · <span class="err">STALLED, partial results</span>' : ''}`;
    document.getElementById('stats').innerHTML =
        stat(d.throughput.toLocaleString(), 'frames / sec', `${d.wall_s}s wall`)
      + stat(d.chips, 'Akida chips used', `${d.tiles} tiles per frame`)
      + stat(d.speedup + '<small>×</small>', 'vs a single chip', `1 chip ≈ ${d.one_chip_rate}/s`)
      + stat(d.avg_seg_ms + '<small>ms</small>', 'on-chip per tile', `+${d.avg_decode_ms}ms decode`)
      + (d.accuracy ? stat(pct(d.accuracy.map50), 'mAP50', `gate ${d.post_thresh}`) : '');
    renderMap(d.accuracy);
    if (d.accuracy_error) { msg.innerHTML += ` · <span class="err">${d.accuracy_error}</span>`; }
    renderGallery(d.gallery);
    // Same numbers onto the node cards, so "which chips ran this" is answered where the
    // fleet already is rather than only in the bar chart below.
    const work = {};
    Object.entries(d.per_host).forEach(([h,v]) =>
      work[h] = {tasks: v.tasks, avg_ms: v.avg_ms, unit: 'tiles'});
    renderFleet(null, work);
    const hosts = Object.entries(d.per_host).sort((a,b)=>a[0].localeCompare(b[0]))
      .map(([h,v]) => [h.replace('.local',''), v.tasks,
                       v.avg_ms + ' ms/tile' + (v.product ? ' · ' + v.product : '')]);
    bars(document.getElementById('hosts'), hosts, ' tiles');
    const cls = Object.entries(d.classes).sort((a,b)=>b[1]-a[1]).slice(0,12);
    document.getElementById('clscard').style.display = cls.length ? '' : 'none';
    bars(document.getElementById('classes'), cls, '');
    document.getElementById('out').style.display='';
  } catch(e) { msg.className='err'; msg.textContent = 'request failed: '+e; }
  btn.disabled = false;
}
document.getElementById('run').onclick = run;
document.getElementById('gt').onchange = () => renderGallery(LAST && LAST.gallery);
loadDatasets();
pollFleet(); setInterval(pollFleet, 5000);
"""

# Concatenated, never %-formatted: the CSS is full of {braces} and the JS of ${literals}
# and `width:...%`, all of which a format string would mangle. See src/common/dashboard_ui.py.
HTML = ('<!doctype html>\n<html><head><meta charset="utf-8">'
        '<title>Symphony + Akida shard inference</title>\n<style>'
        + ui.BASE_CSS + APP_CSS +
        '</style></head><body>\n'
        '<header>\n'
        '  <h1>Symphony + Akida &mdash; image-shard inference</h1>\n'
        '  <p>Each 448&times;448 frame is split into six 224&times;224 tiles &mdash; four'
        ' quadrants, an overlapping centre, and the whole frame downscaled. Symphony fans the'
        ' tiles across the Akida chips (one on-chip inference each) and the stitch service'
        ' merges them back into one result for the frame.</p>\n'
        '</header>\n<main>\n'
        + ui.FLEET_HTML + APP_HTML +
        '</main>\n<script>\n'
        + ui.FLEET_JS + ui.BARS_JS + APP_JS +
        '\n</script>\n</body></html>')

if __name__ == "__main__":
    # threaded (the Flask default) is load-bearing: /api/run blocks this request for the whole
    # run, and the fleet card has to keep polling while it does.
    app.run(host="0.0.0.0", port=PORT, threaded=True)
