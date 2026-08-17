"""One look, one Compute-nodes card, for all three dashboards.

The three apps are three demos of the same fleet, so a viewer should not have to relearn
the page when we switch between them. This module holds what they genuinely share -- the
palette and primitives (BASE_CSS), the node card (FLEET_HTML/FLEET_JS) and the bar rows
(BARS_JS) -- while each dashboard keeps its own panels: model hot-swap for serial-http, the
detection gallery and mAP table for image-shard-inference.

Host-side only, pure stdlib, string constants only. Dashboards assemble their page by
CONCATENATION, never %-formatting or .format(): the CSS is full of {braces} and the JS is
full of ${template literals} and `width:...%`, all of which those two would mangle.

    PAGE = ("<!doctype html>...<style>" + ui.BASE_CSS + APP_CSS + "</style>...<main>"
            + ui.FLEET_HTML + APP_HTML + "</main><script>" + ui.FLEET_JS + APP_JS + "</script>")

FLEET_JS expects an endpoint returning {nodes: [...], summary: {...}, error: null}, where
each node is the record src/common/fleet.py documents. It renders data, never HTML from the
server, so a dashboard cannot inject markup through a node field.
"""

# --- palette + primitives ---------------------------------------------------------------
# The token set image-shard-inference already used (the superset of the three), now shared.
# serial-http was the outlier -- a monospace terminal theme -- and has been ported onto this;
# --mono survives for the places where tabular figures genuinely read better fixed-width.
BASE_CSS = """
  :root { --bg:#0f1420; --card:#1a2130; --line:#26304a; --fg:#e8edf5; --muted:#8b97ad;
          --accent:#4fd1c5; --bar:#3b82f6; --warn:#fbbf24; --good:#34d399; --err:#f87171;
          --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }
  * { box-sizing:border-box; }
  body { margin:0; font:15px/1.5 system-ui,-apple-system,sans-serif;
         background:var(--bg); color:var(--fg); }
  a { color:var(--accent); }
  code { font-family:var(--mono); font-size:.92em; }

  header { padding:20px 28px; border-bottom:1px solid var(--line); }
  header h1 { margin:0; font-size:20px; letter-spacing:-.01em; }
  header p { margin:4px 0 0; color:var(--muted); font-size:13px; max-width:78ch; }
  main { max-width:1100px; margin:0 auto; padding:24px 28px 56px; }

  .controls { display:flex; gap:14px; align-items:flex-end; flex-wrap:wrap;
              background:var(--card); padding:18px; border-radius:12px; }
  label { display:block; font-size:12px; color:var(--muted); margin-bottom:5px; }
  select,input[type=number],input[type=text] {
      background:var(--bg); color:var(--fg); border:1px solid #2b3550; border-radius:8px;
      padding:9px 11px; font:inherit; font-size:14px; }
  button { background:var(--accent); color:#04201d; border:0; border-radius:8px;
           padding:10px 20px; font-weight:600; cursor:pointer; font:inherit; font-size:14px; }
  button:disabled { opacity:.5; cursor:default; }
  button.ghost { background:transparent; color:var(--fg); border:1px solid var(--line);
                 font-weight:500; padding:9px 15px; }
  button.ghost:hover { border-color:var(--accent); }
  button.ghost.danger:hover { border-color:var(--err); color:var(--err); }

  .card { background:var(--card); border-radius:12px; padding:18px 20px; margin-bottom:16px; }
  .card h2 { margin:0 0 14px; font-size:14px; color:var(--muted); font-weight:600;
             text-transform:uppercase; letter-spacing:.04em; }
  .card h2 .aside { text-transform:none; letter-spacing:0; font-weight:400; float:right;
                    color:var(--muted); }

  .stats { display:flex; gap:14px; margin:20px 0; flex-wrap:wrap; }
  .stat { background:var(--card); border-radius:12px; padding:16px 20px; flex:1; min-width:148px; }
  .stat .n { font-size:27px; font-weight:700; font-variant-numeric:tabular-nums;
             letter-spacing:-.02em; }
  .stat .l { color:var(--muted); font-size:12px; margin-top:2px; }
  .stat .n small { font-size:14px; color:var(--muted); font-weight:400; }
  .stat .sub { color:var(--muted); font-size:11px; font-variant-numeric:tabular-nums; }

  .row { display:flex; align-items:center; gap:10px; margin:6px 0; font-size:13px; }
  .row .name { width:150px; color:var(--muted); }
  .row .track { flex:1; background:var(--bg); border-radius:5px; overflow:hidden; height:20px; }
  .row .fill { background:var(--bar); height:100%; border-radius:5px; }
  .row .val { width:200px; text-align:right; font-variant-numeric:tabular-nums; }

  table { width:100%; border-collapse:collapse; }
  th,td { text-align:left; padding:7px 10px; border-bottom:1px solid var(--line); font-size:13px; }
  th { color:var(--muted); font-weight:600; font-size:12px; }
  tr:last-child td { border-bottom:0; }

  .note { margin:16px 0 0; padding:12px 14px; border-radius:10px; font-size:13px;
          border:1px solid var(--line); background:#151c2b; color:var(--muted); }
  .note.warn { border-color:#5a4a12; background:#241d08; color:#f5d78e; }
  .note b { color:var(--fg); }
  .muted { color:var(--muted); }
  .mono { font-family:var(--mono); font-variant-numeric:tabular-nums; }
  .err { color:var(--err); }
  #msg { color:var(--muted); font-size:13px; margin:14px 0; }

  /* --- Compute nodes card ------------------------------------------------------------ */
  .nodes { display:grid; grid-template-columns:repeat(auto-fill,minmax(228px,1fr)); gap:12px; }
  .node { background:var(--bg); border:1px solid var(--line); border-radius:10px;
          padding:11px 13px; }
  .node .who { font-weight:600; display:flex; align-items:center; gap:7px; }
  .node .chip { margin-top:3px; font-family:var(--mono); font-size:12.5px; }
  .node .chip .sep { color:var(--muted); margin:0 5px; }
  .node .work { margin-top:3px; color:var(--muted); font-size:12px;
                font-variant-numeric:tabular-nums; }
  .node .line { margin-top:3px; color:var(--muted); font-size:12px; }
  .node .badges { margin-top:7px; display:flex; gap:5px; flex-wrap:wrap; }
  .dot { width:8px; height:8px; border-radius:50%; flex:none; background:var(--muted); }
  .dot.ready { background:var(--good); }
  .dot.idle  { background:var(--warn); }
  .dot.down  { background:var(--err); }
  .badge { border-radius:20px; padding:1px 9px; font-size:11px; font-weight:700;
           color:#04201d; background:var(--muted); }
  .badge.hw { background:var(--good); }
  .badge.sw { background:var(--warn); }
  .badge.on { background:transparent; color:var(--good); border:1px solid var(--good);
              font-weight:500; }
  .badge.off { background:transparent; color:var(--muted); border:1px solid var(--line);
               font-weight:500; }

  @media (max-width:640px){ .row .name{width:104px} .row .val{width:132px} }
"""

# --- Compute nodes card ------------------------------------------------------------------
FLEET_HTML = """
  <div class="card">
    <h2>Compute nodes<span class="aside" id="fleetsum">…</span></h2>
    <div class="nodes" id="fleetnodes"></div>
  </div>
"""

# renderFleet() is also called after a run with a {host: {tasks, avg_ms, unit}} map, so the
# same cards gain the work each node actually did instead of being redrawn somewhere else.
FLEET_JS = """
let FLEET = {nodes: []}, FLEET_WORK = null;

function fleetBadges(n) {
  return (n.badges || []).map(b =>
    `<span class="badge ${b.kind || ''}">${b.text}</span>`).join('');
}
function fleetCard(n) {
  const work = (FLEET_WORK && FLEET_WORK[n.host]) || null;
  const workLine = work
    ? `${work.tasks.toLocaleString()} ${work.unit || 'tasks'} · ${work.avg_ms} ms`
    : (n.state === 'down' ? 'not running'
       : n.state === 'ready' ? 'ready · no work yet' : 'starting…');
  return `<div class="node">
    <div class="who"><span class="dot ${n.state || ''}"></span>${n.name}</div>
    <div class="chip">${n.product || '-'}<span class="sep">·</span>${n.chip_node || '-'}</div>
    <div class="work">${workLine}</div>
    ${(n.lines || []).map(l => `<div class="line">${l}</div>`).join('')}
    ${(n.badges || []).length ? `<div class="badges">${fleetBadges(n)}</div>` : ''}
  </div>`;
}
function renderFleet(data, work) {
  if (data) FLEET = data;
  if (work !== undefined) FLEET_WORK = work;
  const sum = document.getElementById('fleetsum');
  const box = document.getElementById('fleetnodes');
  if (FLEET.error) { sum.innerHTML = `<span class="err">${FLEET.error}</span>`; }
  else {
    const s = FLEET.summary || {};
    const fams = Object.entries(s.products || {})
      .map(([p, c]) => `${p} ×${c}`).join(' · ');
    sum.textContent = `${s.total || 0} node${s.total === 1 ? '' : 's'}`
      + ` · ${s.ready || 0} on-chip ready` + (fams ? ` · ${fams}` : '');
  }
  box.innerHTML = (FLEET.nodes || []).map(fleetCard).join('')
    || '<div class="muted">no compute nodes found; bring the cluster up with scripts/launch/up.sh</div>';
}
async function pollFleet() {
  try { renderFleet(await (await fetch('/api/fleet')).json()); }
  catch (e) { renderFleet({nodes: FLEET.nodes, summary: FLEET.summary, error: 'fleet: ' + e}); }
}
"""

# The per-chip distribution rows, previously copy-pasted in two dashboards.
BARS_JS = """
function bars(el, entries, unit) {
  const max = Math.max(1, ...entries.map(e => e[1]));
  el.innerHTML = entries.map(([name, val, extra]) =>
    `<div class="row"><div class="name">${name}</div>
     <div class="track"><div class="fill" style="width:${100*val/max}%"></div></div>
     <div class="val">${val.toLocaleString()}${unit||''}${extra?(' &middot; '+extra):''}</div></div>`).join('');
}
"""
