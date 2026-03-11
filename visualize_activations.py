#!/usr/bin/env python3
"""
🦥 Unsloth Activation Visualizer

Reads the output of --capture_activations (metadata.json + activation_log.jsonl)
and generates a self-contained HTML file you can open in any browser.

The HTML shows a 3D-style animated grid of cubes.  Each column of cubes is one
transformer layer.  Each cube is one tracked hidden-state channel.  Color encodes
the mean absolute activation for that channel at that training step:

    cool blue  →  low / dormant
    warm red   →  high / active
    intensity  →  change relative to the pre-finetune baseline (step 0)

The animation can be stepped through manually or played as a timed slideshow,
making it easy to see *where* in the model the finetuning burns in.

Usage:
    python visualize_activations.py <activation_dir> [--output viz.html] [--open]

    activation_dir   Directory containing metadata.json + activation_log.jsonl
                     (the value you passed to --capture_output_dir in the CLI).
    --output         Path for the generated HTML file  (default: <dir>/viz.html)
    --open           Open the HTML in the default browser after generating it.

Example (after running the CLI with --capture_activations):
    python visualize_activations.py activation_logs --open
"""

import argparse
import json
import os
import sys


# ---------------------------------------------------------------------------
# Load capture data
# ---------------------------------------------------------------------------

def load_data(activation_dir: str):
    meta_path = os.path.join(activation_dir, "metadata.json")
    log_path  = os.path.join(activation_dir, "activation_log.jsonl")

    if not os.path.exists(meta_path):
        sys.exit(f"[error] metadata.json not found in '{activation_dir}'.\n"
                 f"        Run unsloth-cli.py with --capture_activations first.")
    if not os.path.exists(log_path):
        sys.exit(f"[error] activation_log.jsonl not found in '{activation_dir}'.")

    with open(meta_path) as f:
        metadata = json.load(f)

    records = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        sys.exit("[error] activation_log.jsonl is empty — no steps were captured.")

    print(f"Loaded {len(records)} captured steps from '{activation_dir}'.")
    return metadata, records


# ---------------------------------------------------------------------------
# Build the HTML
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>🦥 Unsloth – Neuron Activation Map</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: #0d1117;
    color: #e6edf3;
    font-family: 'Segoe UI', system-ui, sans-serif;
    display: flex;
    flex-direction: column;
    align-items: center;
    min-height: 100vh;
    padding: 24px 16px;
  }}
  h1 {{ font-size: 1.4rem; font-weight: 600; margin-bottom: 4px; }}
  .subtitle {{ font-size: 0.82rem; color: #8b949e; margin-bottom: 20px; }}
  #canvas-wrap {{
    border: 1px solid #30363d;
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 4px 24px rgba(0,0,0,.5);
  }}
  canvas {{ display: block; }}
  .controls {{
    display: flex;
    align-items: center;
    gap: 12px;
    margin-top: 16px;
    flex-wrap: wrap;
    justify-content: center;
  }}
  button {{
    background: #21262d;
    color: #e6edf3;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 6px 14px;
    cursor: pointer;
    font-size: 0.85rem;
  }}
  button:hover {{ background: #2d333b; }}
  button.active {{ border-color: #58a6ff; color: #58a6ff; }}
  #step-label {{ font-size: 0.82rem; color: #8b949e; min-width: 160px; text-align: center; }}
  input[type=range] {{ width: 260px; accent-color: #58a6ff; }}
  .legend {{
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 14px;
    font-size: 0.78rem;
    color: #8b949e;
  }}
  .legend-bar {{
    width: 140px;
    height: 12px;
    border-radius: 4px;
    background: linear-gradient(to right, #1e3a5f, #2660a4, #e8a838, #c0392b);
  }}
  .info-row {{
    font-size: 0.78rem;
    color: #8b949e;
    margin-top: 8px;
    text-align: center;
  }}
</style>
</head>
<body>
<h1>🦥 Unsloth Neuron Activation Map</h1>
<p class="subtitle" id="model-label">Loading…</p>

<div id="canvas-wrap">
  <canvas id="c"></canvas>
</div>

<div class="controls">
  <button id="btn-play">▶ Play</button>
  <input type="range" id="slider" min="0" value="0">
  <span id="step-label">Step 0</span>
</div>
<div class="controls">
  <button id="btn-diff">Show: absolute</button>
  <button id="btn-slower">slower</button>
  <button id="btn-faster">faster</button>
</div>
<div class="legend">
  <span>low</span>
  <div class="legend-bar"></div>
  <span>high</span>
</div>
<p class="info-row" id="info-row">&nbsp;</p>

<script>
// ---- embedded data -------------------------------------------------------
const META    = {meta_json};
const RECORDS = {records_json};
// --------------------------------------------------------------------------

const canvas = document.getElementById('c');
const ctx    = canvas.getContext('2d');

const N_LAYERS   = META.num_layers;
const CHANNELS   = META.captured_channels.length;
const CELL       = 10;   // px per neuron cell
const GAP        = 2;    // px between cells
const LAYER_GAP  = 6;    // extra px between layers
const PAD        = 40;   // canvas padding
const LABEL_W    = 36;   // space for layer labels
const TOP_PAD    = 50;   // space for step/loss info at top

const colW  = CELL + GAP;
const rowH  = CELL + GAP;
const gridW = N_LAYERS * colW + (N_LAYERS - 1) * (LAYER_GAP - GAP) + LABEL_W;
const gridH = CHANNELS * rowH;

canvas.width  = gridW + PAD * 2;
canvas.height = gridH + PAD * 2 + TOP_PAD;

document.getElementById('model-label').textContent =
    (META.model_name || 'unknown model') +
    '  ·  ' + N_LAYERS + ' layers  ·  ' + CHANNELS + ' channels';

const slider = document.getElementById('slider');
slider.max   = RECORDS.length - 1;

// ---- colour mapping (cool→warm heat scale) --------------------------------
function lerp(a, b, t) {{ return a + (b - a) * t; }}
function heatColor(v) {{
  // v in [0,1] → cool blue to warm red
  // break into 3 stops: [0] #1e3a5f,  [0.5] #e8a838,  [1] #c0392b
  let r, g, b;
  if (v < 0.5) {{
    const t = v * 2;
    r = Math.round(lerp(0x1e, 0xe8, t));
    g = Math.round(lerp(0x3a, 0xa8, t));
    b = Math.round(lerp(0x5f, 0x38, t));
  }} else {{
    const t = (v - 0.5) * 2;
    r = Math.round(lerp(0xe8, 0xc0, t));
    g = Math.round(lerp(0xa8, 0x39, t));
    b = Math.round(lerp(0x38, 0x2b, t));
  }}
  return `rgb(${{r}},${{g}},${{b}})`;
}}

// ---- pre-compute per-record normalised values ----------------------------
// Use the first record as the baseline (step 0, pre-finetune).
let showDiff = false;

function getValues(record, layerIdx) {{
  const key = String(layerIdx);
  return record.layers[key] ? record.layers[key].mean_abs : null;
}}

// Global max across all records / all layers (for stable colour scale).
let globalMax = 1e-9;
for (const rec of RECORDS) {{
  for (let l = 0; l < N_LAYERS; l++) {{
    const vals = getValues(rec, l);
    if (!vals) continue;
    for (const v of vals) {{ if (v > globalMax) globalMax = v; }}
  }}
}}

function baselineVals(layerIdx) {{
  return getValues(RECORDS[0], layerIdx);
}}

function normalisedValues(record, layerIdx) {{
  const vals = getValues(record, layerIdx);
  if (!vals) return null;
  if (!showDiff) {{
    return vals.map(v => v / globalMax);
  }}
  // diff mode: show change relative to baseline
  const base = baselineVals(layerIdx);
  return vals.map((v, i) => {{
    const diff = v - (base ? base[i] : 0);
    // map [-globalMax, +globalMax] -> [0, 1]
    return Math.max(0, Math.min(1, (diff / globalMax) * 0.5 + 0.5));
  }});
}}

// ---- draw -----------------------------------------------------------------
function draw(idx) {{
  const rec  = RECORDS[idx];
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // background
  ctx.fillStyle = '#0d1117';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // step / loss header
  ctx.fillStyle = '#8b949e';
  ctx.font = '12px monospace';
  const loss = rec.loss != null ? ('loss ' + rec.loss.toFixed(4)) : '';
  ctx.fillText(`step ${{rec.step}}  ${{loss}}`, PAD + LABEL_W, PAD + 18);

  // layer column headers
  ctx.font = '10px monospace';
  ctx.fillStyle = '#4d8bf0';
  for (let l = 0; l < N_LAYERS; l++) {{
    const x = PAD + LABEL_W + l * (colW + LAYER_GAP - GAP);
    const y = PAD + TOP_PAD - 6;
    ctx.save();
    ctx.translate(x + CELL / 2, y);
    ctx.rotate(-Math.PI / 3);
    ctx.fillText(l, 0, 0);
    ctx.restore();
  }}

  // channel row labels (every 8)
  ctx.fillStyle = '#8b949e';
  ctx.font      = '9px monospace';
  for (let c = 0; c < CHANNELS; c += 8) {{
    const y = PAD + TOP_PAD + c * rowH + CELL;
    ctx.fillText(META.captured_channels[c], PAD, y);
  }}

  // neuron cells
  for (let l = 0; l < N_LAYERS; l++) {{
    const x0 = PAD + LABEL_W + l * (colW + LAYER_GAP - GAP);
    const nv = normalisedValues(rec, l);
    for (let c = 0; c < CHANNELS; c++) {{
      const x = x0;
      const y = PAD + TOP_PAD + c * rowH;
      const v = nv ? nv[c] : 0;
      ctx.fillStyle = heatColor(v);
      ctx.beginPath();
      ctx.roundRect(x, y, CELL, CELL, 2);
      ctx.fill();
    }}
  }}

  // update UI
  document.getElementById('step-label').textContent =
    `step ${{rec.step}}${{rec.loss != null ? '  loss ' + rec.loss.toFixed(4) : ''}}`;
}}

draw(0);

// ---- playback -----------------------------------------------------------
let playing  = false;
let frameMs  = 300;
let timer    = null;
let curIdx   = 0;

function setIdx(i) {{
  curIdx = Math.max(0, Math.min(RECORDS.length - 1, i));
  slider.value = curIdx;
  draw(curIdx);
}}

function tick() {{
  if (!playing) return;
  setIdx(curIdx + 1);
  if (curIdx >= RECORDS.length - 1) {{ stopPlay(); return; }}
  timer = setTimeout(tick, frameMs);
}}

function startPlay() {{
  playing = true;
  document.getElementById('btn-play').textContent = '⏸ Pause';
  document.getElementById('btn-play').classList.add('active');
  timer = setTimeout(tick, frameMs);
}}

function stopPlay() {{
  playing = false;
  clearTimeout(timer);
  document.getElementById('btn-play').textContent = '▶ Play';
  document.getElementById('btn-play').classList.remove('active');
}}

document.getElementById('btn-play').onclick = () => {{
  if (playing) stopPlay();
  else {{ if (curIdx >= RECORDS.length - 1) setIdx(0); startPlay(); }}
}};
slider.oninput = () => {{ stopPlay(); setIdx(+slider.value); }};
document.getElementById('btn-slower').onclick = () => {{ frameMs = Math.min(2000, frameMs + 100); }};
document.getElementById('btn-faster').onclick = () => {{ frameMs = Math.max(50,  frameMs - 100); }};
document.getElementById('btn-diff').onclick = function() {{
  showDiff = !showDiff;
  this.textContent = showDiff ? 'Show: diff vs step 0' : 'Show: absolute';
  draw(curIdx);
}};
</script>
</body>
</html>
"""


def build_html(metadata: dict, records: list) -> str:
    meta_json    = json.dumps(metadata, separators=(",", ":"))
    records_json = json.dumps(records,  separators=(",", ":"))
    return _HTML_TEMPLATE.format(meta_json=meta_json, records_json=records_json)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="🦥 Unsloth – Generate a browser-based neuron activation animation"
    )
    parser.add_argument(
        "activation_dir",
        help="Directory containing metadata.json + activation_log.jsonl",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output HTML path (default: <activation_dir>/viz.html)",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the generated HTML in the default browser",
    )
    args = parser.parse_args()

    metadata, records = load_data(args.activation_dir)
    html = build_html(metadata, records)

    out_path = args.output or os.path.join(args.activation_dir, "viz.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    size_kb = os.path.getsize(out_path) / 1024
    print(f"✅  Visualization written to '{out_path}'  ({size_kb:.1f} KB)")
    print(f"    Open it in any browser — no server required.\n")

    if args.open:
        import webbrowser
        webbrowser.open(f"file://{os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
