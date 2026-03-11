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
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Unsloth Activation Viewer</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400&display=swap" rel="stylesheet">
<style>
:root {{
  --bg: #08080c;
  --bg-card: #0e0e14;
  --border: rgba(255,255,255,0.08);
  --text: #e8e8f0;
  --text2: #888894;
  --text3: #555560;
  --blue: #6366f1;
  --cyan: #22d3ee;
  --green: #10b981;
  --purple: #a855f7;
  --orange: #f59e0b;
  --red: #ef4444;
}}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
  background: var(--bg);
  color: var(--text);
  font-family: 'Inter', system-ui, sans-serif;
  min-height: 100vh;
  padding: 20px;
}}
.top-bar {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 20px;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 12px;
  margin-bottom: 16px;
}}
.brand {{
  display: flex;
  align-items: center;
  gap: 10px;
  font-weight: 600;
  font-size: 1rem;
}}
.brand-icon {{ font-size: 1.4rem; }}
.model-info {{
  font-size: 0.75rem;
  color: var(--text2);
  font-family: 'JetBrains Mono', monospace;
}}
.view-toggle {{
  display: flex;
  gap: 4px;
  background: var(--bg);
  padding: 4px;
  border-radius: 8px;
}}
.view-btn {{
  padding: 6px 16px;
  border: none;
  background: transparent;
  color: var(--text2);
  font-family: inherit;
  font-size: 0.8rem;
  font-weight: 500;
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.15s;
}}
.view-btn.active {{
  background: var(--blue);
  color: white;
}}
.main-layout {{
  display: grid;
  grid-template-columns: 1fr 280px;
  gap: 16px;
  height: calc(100vh - 140px);
}}
.viz-area {{
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 12px;
  overflow: hidden;
  position: relative;
}}
.side-panel {{
  display: flex;
  flex-direction: column;
  gap: 12px;
}}
.card {{
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 14px;
}}
.card-header {{
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 10px;
  font-size: 0.75rem;
  font-weight: 600;
  color: var(--text2);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}}
.card-icon {{
  width: 20px;
  height: 20px;
  border-radius: 5px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.7rem;
}}
.card-icon.grad {{ background: rgba(16,185,129,0.15); color: var(--green); }}
.card-icon.lora {{ background: rgba(168,85,247,0.15); color: var(--purple); }}
canvas {{ display: block; }}
.grad-row, .lora-row {{
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 6px;
}}
.grad-row:last-child, .lora-row:last-child {{ margin-bottom: 0; }}
.metric-label {{
  font-size: 0.65rem;
  color: var(--text3);
  font-family: 'JetBrains Mono', monospace;
  width: 32px;
  flex-shrink: 0;
}}
.spark-wrap {{
  flex: 1;
  height: 18px;
  background: var(--bg);
  border-radius: 4px;
  overflow: hidden;
}}
.spark-bar {{
  height: 100%;
  border-radius: 4px;
  transition: width 0.1s;
}}
.spark-bar.grad {{ background: linear-gradient(90deg, #064e3b, var(--green)); }}
.spark-bar.lora {{ background: linear-gradient(90deg, #4c1d95, var(--purple)); }}
.controls {{
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 14px;
}}
.play-btn {{
  width: 36px;
  height: 36px;
  border-radius: 50%;
  border: 1px solid var(--border);
  background: var(--bg);
  color: var(--text);
  font-size: 0.9rem;
  cursor: pointer;
  transition: all 0.15s;
}}
.play-btn:hover {{ border-color: var(--blue); }}
.play-btn.active {{ background: var(--blue); border-color: var(--blue); }}
.slider-area {{ flex: 1; }}
.slider-row {{
  display: flex;
  align-items: center;
  gap: 8px;
}}
input[type=range] {{
  flex: 1;
  height: 6px;
  -webkit-appearance: none;
  background: var(--bg);
  border-radius: 3px;
  cursor: pointer;
}}
input[type=range]::-webkit-slider-thumb {{
  -webkit-appearance: none;
  width: 14px;
  height: 14px;
  border-radius: 50%;
  background: var(--text);
  box-shadow: 0 0 0 2px var(--blue);
}}
.step-display {{
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.7rem;
  color: var(--text2);
  min-width: 80px;
  text-align: right;
}}
.opts {{
  display: flex;
  gap: 6px;
}}
.opt-btn {{
  padding: 5px 10px;
  border: 1px solid var(--border);
  background: transparent;
  color: var(--text2);
  font-size: 0.7rem;
  font-family: inherit;
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.15s;
}}
.opt-btn:hover {{ border-color: var(--blue); color: var(--text); }}
.opt-btn.active {{ background: rgba(99,102,241,0.15); border-color: var(--blue); color: var(--blue); }}
.canvas-2d {{ width: 100%; height: calc(100% - 60px); }}
#c2d {{ width: 100%; height: 100%; }}
.view-3d {{
  width: 100%;
  height: calc(100% - 60px);
  perspective: 1200px;
  overflow: hidden;
  display: none;
}}
.cube-scene {{
  width: 100%;
  height: 100%;
  transform-style: preserve-3d;
  display: flex;
  align-items: center;
  justify-content: center;
}}
.prism {{
  transform-style: preserve-3d;
  transform: rotateX(-25deg) rotateY(-35deg);
}}
.slice {{
  position: absolute;
  transform-style: preserve-3d;
  transition: transform 0.3s ease-out;
}}
.cube {{
  position: absolute;
  width: 8px;
  height: 8px;
  transform-style: preserve-3d;
}}
.cube-face {{
  position: absolute;
  width: 8px;
  height: 8px;
  backface-visibility: hidden;
}}
.cube-face.front {{ transform: translateZ(4px); }}
.cube-face.back {{ transform: rotateY(180deg) translateZ(4px); }}
.cube-face.top {{ transform: rotateX(90deg) translateZ(4px); }}
.cube-face.bottom {{ transform: rotateX(-90deg) translateZ(4px); }}
.cube-face.left {{ transform: rotateY(-90deg) translateZ(4px); }}
.cube-face.right {{ transform: rotateY(90deg) translateZ(4px); }}
.slice-control {{
  position: absolute;
  bottom: 70px;
  left: 50%;
  transform: translateX(-50%);
  display: flex;
  align-items: center;
  gap: 12px;
  background: rgba(14,14,20,0.9);
  padding: 10px 16px;
  border-radius: 10px;
  border: 1px solid var(--border);
  z-index: 10;
}}
.slice-label {{
  font-size: 0.7rem;
  color: var(--text2);
}}
#slice-slider {{
  width: 200px;
}}
.legend {{
  display: flex;
  gap: 16px;
  justify-content: center;
  padding: 8px;
  font-size: 0.65rem;
  color: var(--text3);
}}
.legend-item {{
  display: flex;
  align-items: center;
  gap: 6px;
}}
.legend-bar {{
  width: 60px;
  height: 6px;
  border-radius: 3px;
}}
.legend-bar.act {{ background: linear-gradient(90deg, #1e3a5f, #3b82f6, #f59e0b, #ef4444); }}
.hidden {{ display: none !important; }}
    border: 1px solid var(--border-subtle);
    border-radius: 20px;
    padding: 6px 14px;
    margin-top: 12px;
    font-size: 0.75rem;
    color: var(--text-secondary);
    backdrop-filter: blur(10px);
  }}
  .model-badge .dot {{
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--accent-green);
    box-shadow: 0 0 8px var(--accent-green);
  }}
  .panels {{
    display: flex;
    gap: 20px;
    flex-wrap: wrap;
    justify-content: center;
    margin-bottom: 28px;
  }}
  .panel {{
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 16px;
    padding: 20px;
    backdrop-filter: blur(20px);
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
  }}
  .panel::before {{
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
  }}
  .panel:hover {{
    border-color: var(--border-accent);
    box-shadow: 0 8px 32px rgba(99, 102, 241, 0.1);
  }}
  .panel-header {{
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 14px;
  }}
  .panel-icon {{
    width: 28px;
    height: 28px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.875rem;
  }}
  .panel-icon.act {{ background: var(--glow-blue); color: var(--accent-blue); }}
  .panel-icon.grad {{ background: var(--glow-green); color: var(--accent-green); }}
  .panel-icon.lora {{ background: var(--glow-purple); color: var(--accent-purple); }}
  .panel-title {{
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--text-primary);
    letter-spacing: 0.02em;
  }}
  .panel-subtitle {{
    font-size: 0.7rem;
    color: var(--text-muted);
    margin-left: auto;
    font-family: 'JetBrains Mono', monospace;
  }}
  .canvas-wrap {{
    border-radius: 10px;
    overflow: hidden;
    background: var(--bg-primary);
  }}
  canvas {{ display: block; }}
  .controls-section {{
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 16px;
    padding: 20px 28px;
    backdrop-filter: blur(20px);
    max-width: 700px;
    margin: 0 auto 20px;
  }}
  .playback-row {{
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 16px;
  }}
  .btn {{
    background: var(--bg-secondary);
    color: var(--text-primary);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    padding: 10px 18px;
    cursor: pointer;
    font-size: 0.875rem;
    font-weight: 500;
    font-family: inherit;
    transition: all 0.2s ease;
    display: inline-flex;
    align-items: center;
    gap: 6px;
  }}
  .btn:hover {{
    background: rgba(99, 102, 241, 0.1);
    border-color: var(--accent-blue);
  }}
  .btn.active {{
    background: rgba(99, 102, 241, 0.15);
    border-color: var(--accent-blue);
    color: var(--accent-blue);
    box-shadow: 0 0 20px var(--glow-blue);
  }}
  .btn-icon {{
    width: 36px;
    height: 36px;
    padding: 0;
    border-radius: 50%;
    font-size: 1rem;
  }}
  .slider-wrap {{
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }}
  .slider-track {{
    position: relative;
    height: 6px;
    background: var(--bg-primary);
    border-radius: 3px;
    overflow: hidden;
  }}
  .slider-progress {{
    position: absolute;
    top: 0;
    left: 0;
    height: 100%;
    background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan));
    border-radius: 3px;
    transition: width 0.1s ease;
  }}
  input[type=range] {{
    width: 100%;
    height: 6px;
    -webkit-appearance: none;
    background: transparent;
    position: relative;
    z-index: 2;
    margin: 0;
  }}
  input[type=range]::-webkit-slider-thumb {{
    -webkit-appearance: none;
    width: 16px;
    height: 16px;
    border-radius: 50%;
    background: var(--text-primary);
    cursor: pointer;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3), 0 0 0 3px var(--accent-blue);
    transition: transform 0.15s ease;
    margin-top: -5px;
  }}
  input[type=range]::-webkit-slider-thumb:hover {{
    transform: scale(1.15);
  }}
  input[type=range]::-webkit-slider-runnable-track {{
    height: 6px;
    background: transparent;
  }}
  .step-info {{
    display: flex;
    justify-content: space-between;
    font-size: 0.75rem;
  }}
  .step-label {{
    font-family: 'JetBrains Mono', monospace;
    color: var(--text-primary);
  }}
  .loss-label {{
    color: var(--accent-cyan);
  }}
  .speed-label {{
    color: var(--text-muted);
  }}
  .options-row {{
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 10px;
    padding-top: 12px;
    border-top: 1px solid var(--border-subtle);
  }}
  .btn-sm {{
    padding: 6px 12px;
    font-size: 0.75rem;
    border-radius: 8px;
  }}
  .legends {{
    display: flex;
    justify-content: center;
    gap: 24px;
    flex-wrap: wrap;
    margin-bottom: 16px;
  }}
  .legend {{
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.75rem;
    color: var(--text-secondary);
  }}
  .legend-label {{
    font-weight: 500;
    color: var(--text-muted);
  }}
  .legend-bar {{
    width: 100px;
    height: 8px;
    border-radius: 4px;
  }}
  .legend-bar.act {{
    background: linear-gradient(90deg, #1e3a5f, #3b82f6, #f59e0b, #ef4444);
  }}
  .legend-bar.grad {{
    background: linear-gradient(90deg, var(--bg-primary), #059669, var(--accent-green));
  }}
  .legend-bar.lora {{
    background: linear-gradient(90deg, var(--bg-primary), #7c3aed, var(--accent-purple));
  }}
  .shortcuts {{
    text-align: center;
    font-size: 0.7rem;
    color: var(--text-muted);
  }}
  .shortcuts kbd {{
    display: inline-block;
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 4px;
    padding: 2px 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    margin: 0 2px;
  }}
  @keyframes pulse {{
    0%, 100% {{ opacity: 1; }}
    50% {{ opacity: 0.5; }}
  }}
  .recording-dot {{
    animation: pulse 1.5s ease-in-out infinite;
  }}
</style>
</head>
<body>
<div class="container">
  <header>
    <div class="logo">
      <span class="logo-icon">🦥</span>
      <h1>Unsloth</h1>
    </div>
    <p class="subtitle">Activation Viewer</p>
    <div class="model-badge" id="model-label">
      <span class="dot"></span>
      <span>Loading model...</span>
    </div>
  </header>

  <div class="panels">
    <div class="panel" id="act-panel">
      <div class="panel-header">
        <div class="panel-icon act">⚡</div>
        <span class="panel-title">Activations</span>
        <span class="panel-subtitle" id="act-dims"></span>
      </div>
      <div class="canvas-wrap" id="canvas-wrap">
        <canvas id="c"></canvas>
      </div>
    </div>
    <div class="panel" id="grad-panel" style="display:none;">
      <div class="panel-header">
        <div class="panel-icon grad">∇</div>
        <span class="panel-title">Gradient Norms</span>
        <span class="panel-subtitle">learning signal</span>
      </div>
      <div class="canvas-wrap" id="grad-canvas-wrap">
        <canvas id="grad-c"></canvas>
      </div>
    </div>
    <div class="panel" id="lora-panel" style="display:none;">
      <div class="panel-header">
        <div class="panel-icon lora">◇</div>
        <span class="panel-title">LoRA Growth</span>
        <span class="panel-subtitle">||B·A||<sub>F</sub></span>
      </div>
      <div class="canvas-wrap" id="lora-canvas-wrap">
        <canvas id="lora-c"></canvas>
      </div>
    </div>
  </div>

  <div class="controls-section">
    <div class="playback-row">
      <button class="btn btn-icon" id="btn-play" title="Play/Pause (Space)">▶</button>
      <div class="slider-wrap">
        <div class="slider-track">
          <div class="slider-progress" id="slider-progress"></div>
          <input type="range" id="slider" min="0" value="0">
        </div>
        <div class="step-info">
          <span class="step-label" id="step-label">Step 0</span>
          <span class="loss-label" id="loss-label"></span>
          <span class="speed-label" id="speed-label">1.0×</span>
        </div>
      </div>
    </div>
    <div class="options-row">
      <button class="btn btn-sm" id="btn-diff">Absolute</button>
      <button class="btn btn-sm" id="btn-slower">−</button>
      <button class="btn btn-sm" id="btn-faster">+</button>
    </div>
  </div>

  <div class="legends">
    <div class="legend" id="legend-act">
      <span class="legend-label">Activation</span>
      <span>low</span>
      <div class="legend-bar act"></div>
      <span>high</span>
    </div>
    <div class="legend" id="legend-grad" style="display:none;">
      <span class="legend-label">Gradient</span>
      <span>idle</span>
      <div class="legend-bar grad"></div>
      <span>learning</span>
    </div>
    <div class="legend" id="legend-lora" style="display:none;">
      <span class="legend-label">LoRA</span>
      <span>zero</span>
      <div class="legend-bar lora"></div>
      <span>high</span>
    </div>
  </div>

  <div class="shortcuts">
    <kbd>Space</kbd> Play/Pause &nbsp; <kbd>←</kbd><kbd>→</kbd> Step &nbsp; <kbd>D</kbd> Toggle diff
  </div>
</div>

<script>
// ---- embedded data -------------------------------------------------------
const META    = {meta_json};
const RECORDS = {records_json};
// --------------------------------------------------------------------------

// Check if we have gradient and LoRA data
const HAS_GRADS = RECORDS.some(r => r.grad_norms && Object.keys(r.grad_norms).length > 0);
const HAS_LORA  = RECORDS.some(r => r.lora_norms && Object.keys(r.lora_norms).length > 0);

// Show/hide panels based on data availability
if (HAS_GRADS) {{
  document.getElementById('grad-panel').style.display = 'block';
  document.getElementById('legend-grad').style.display = 'flex';
}}
if (HAS_LORA) {{
  document.getElementById('lora-panel').style.display = 'block';
  document.getElementById('legend-lora').style.display = 'flex';
}}

const canvas = document.getElementById('c');
const ctx    = canvas.getContext('2d');
const gradCanvas = document.getElementById('grad-c');
const gradCtx = gradCanvas ? gradCanvas.getContext('2d') : null;
const loraCanvas = document.getElementById('lora-c');
const loraCtx = loraCanvas ? loraCanvas.getContext('2d') : null;

const N_LAYERS   = META.num_layers;
const CHANNELS   = META.captured_channels.length;
const CELL       = 11;   // px per neuron cell
const GAP        = 2;    // px between cells
const LAYER_GAP  = 5;    // extra px between layers
const PAD        = 44;   // canvas padding
const LABEL_W    = 40;   // space for layer labels
const TOP_PAD    = 36;   // space for header

const colW  = CELL + GAP;
const rowH  = CELL + GAP;
const gridW = N_LAYERS * colW + (N_LAYERS - 1) * (LAYER_GAP - GAP) + LABEL_W;
const gridH = CHANNELS * rowH;

canvas.width  = gridW + PAD * 2;
canvas.height = gridH + PAD + TOP_PAD;

// Update dimensions display
document.getElementById('act-dims').textContent = `${{N_LAYERS}}L × ${{CHANNELS}}ch`;

// Gradient canvas: single row per layer (bar chart style)
const GRAD_BAR_H = 28;
const GRAD_H = GRAD_BAR_H + TOP_PAD + PAD + 20;
if (gradCanvas) {{
  gradCanvas.width = gridW + PAD * 2;
  gradCanvas.height = GRAD_H;
}}

// LoRA canvas: depends on number of targets
const loraTargets = META.lora_targets || [];
const LORA_ROW_H = 24;
const LORA_H = loraTargets.length * (LORA_ROW_H + 4) + TOP_PAD + PAD;
if (loraCanvas && loraTargets.length > 0) {{
  loraCanvas.width = gridW + PAD * 2;
  loraCanvas.height = LORA_H;
}}

// Update model badge
const modelName = (META.model_name || 'unknown').split('/').pop();
document.getElementById('model-label').innerHTML = `
  <span class="dot"></span>
  <span>${{modelName}}</span>
  <span style="color:var(--text-muted)">·</span>
  <span>${{N_LAYERS}} layers</span>
  <span style="color:var(--text-muted)">·</span>
  <span>${{CHANNELS}} channels</span>
  ${{HAS_GRADS ? '<span style="color:var(--text-muted)">·</span><span style="color:var(--accent-green)">∇</span>' : ''}}
  ${{HAS_LORA ? '<span style="color:var(--text-muted)">·</span><span style="color:var(--accent-purple)">◇</span>' : ''}}
`;

const slider = document.getElementById('slider');
const sliderProgress = document.getElementById('slider-progress');
slider.max   = RECORDS.length - 1;

// ---- colour mapping -------------------------------------------------------
function lerp(a, b, t) {{ return a + (b - a) * t; }}

function heatColor(v) {{
  // More vibrant color scale: deep blue → cyan → yellow → red
  let r, g, b;
  if (v < 0.33) {{
    const t = v * 3;
    r = Math.round(lerp(0x1a, 0x22, t));
    g = Math.round(lerp(0x2a, 0xd3, t));
    b = Math.round(lerp(0x6c, 0xee, t));
  }} else if (v < 0.66) {{
    const t = (v - 0.33) * 3;
    r = Math.round(lerp(0x22, 0xf5, t));
    g = Math.round(lerp(0xd3, 0x9e, t));
    b = Math.round(lerp(0xee, 0x0b, t));
  }} else {{
    const t = (v - 0.66) * 3;
    r = Math.round(lerp(0xf5, 0xef, t));
    g = Math.round(lerp(0x9e, 0x44, t));
    b = Math.round(lerp(0x0b, 0x44, t));
  }}
  return `rgb(${{r}},${{g}},${{b}})`;
}}

function gradColor(v) {{
  // Dark → emerald → bright green
  const r = Math.round(lerp(0x0a, 0x10, Math.min(1, v * 2)) + lerp(0, 0x20, Math.max(0, v - 0.5) * 2));
  const g = Math.round(lerp(0x0a, 0xb9, v));
  const b = Math.round(lerp(0x0f, 0x81, v));
  return `rgb(${{r}},${{g}},${{b}})`;
}}

function loraColor(v) {{
  // Dark → violet → bright purple
  const r = Math.round(lerp(0x0a, 0xa8, v));
  const g = Math.round(lerp(0x0a, 0x55, v));
  const b = Math.round(lerp(0x0f, 0xf7, v));
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

// ---- Gradient norms max for normalization --------------------------------
let globalGradMax = 1e-9;
for (const rec of RECORDS) {{
  if (!rec.grad_norms) continue;
  for (const v of Object.values(rec.grad_norms)) {{
    if (v > globalGradMax) globalGradMax = v;
  }}
}}

// ---- LoRA norms max for normalization ------------------------------------
let globalLoraMax = 1e-9;
for (const rec of RECORDS) {{
  if (!rec.lora_norms) continue;
  for (const v of Object.values(rec.lora_norms)) {{
    if (v > globalLoraMax) globalLoraMax = v;
  }}
}}

// ---- draw activation heatmap ---------------------------------------------
function drawActivations(idx) {{
  const rec  = RECORDS[idx];
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // background
  ctx.fillStyle = '#0a0a0f';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // layer column headers (every 4)
  ctx.font = '500 10px "JetBrains Mono", monospace';
  ctx.fillStyle = '#6366f1';
  for (let l = 0; l < N_LAYERS; l += 4) {{
    const x = PAD + LABEL_W + l * (colW + LAYER_GAP - GAP);
    ctx.fillText(l, x + CELL / 2 - 4, PAD + 12);
  }}

  // channel row labels (every 8)
  ctx.fillStyle = '#5c5c6e';
  ctx.font = '10px "JetBrains Mono", monospace';
  for (let c = 0; c < CHANNELS; c += 8) {{
    const y = TOP_PAD + PAD + c * rowH + CELL - 1;
    ctx.fillText(META.captured_channels[c], 8, y);
  }}

  // neuron cells with rounded corners
  for (let l = 0; l < N_LAYERS; l++) {{
    const x0 = PAD + LABEL_W + l * (colW + LAYER_GAP - GAP);
    const nv = normalisedValues(rec, l);
    for (let c = 0; c < CHANNELS; c++) {{
      const x = x0;
      const y = TOP_PAD + PAD + c * rowH;
      const v = nv ? nv[c] : 0;
      ctx.fillStyle = heatColor(v);
      ctx.beginPath();
      ctx.roundRect(x, y, CELL, CELL, 3);
      ctx.fill();
    }}
  }}
}}

// ---- draw gradient norms (bar chart per layer) ---------------------------
function drawGradients(idx) {{
  if (!gradCtx || !HAS_GRADS) return;
  const rec = RECORDS[idx];
  gradCtx.clearRect(0, 0, gradCanvas.width, gradCanvas.height);
  gradCtx.fillStyle = '#0a0a0f';
  gradCtx.fillRect(0, 0, gradCanvas.width, gradCanvas.height);

  // bars per layer
  const barW = (gradCanvas.width - PAD * 2 - LABEL_W) / N_LAYERS - 2;
  const maxBarH = GRAD_BAR_H;
  
  for (let l = 0; l < N_LAYERS; l++) {{
    const gradVal = rec.grad_norms ? (rec.grad_norms[String(l)] || 0) : 0;
    const normVal = Math.min(1, gradVal / globalGradMax);
    const x = PAD + LABEL_W + l * (barW + 2);
    const barH = maxBarH * normVal;
    const y = TOP_PAD + maxBarH - barH;

    // Glow effect for high values
    if (normVal > 0.5) {{
      gradCtx.shadowColor = '#10b981';
      gradCtx.shadowBlur = normVal * 8;
    }} else {{
      gradCtx.shadowBlur = 0;
    }}

    gradCtx.fillStyle = gradColor(normVal);
    gradCtx.beginPath();
    gradCtx.roundRect(x, y, barW - 1, barH || 2, 2);
    gradCtx.fill();
    gradCtx.shadowBlur = 0;

    // layer label (every 4)
    if (l % 4 === 0) {{
      gradCtx.fillStyle = '#5c5c6e';
      gradCtx.font = '9px "JetBrains Mono", monospace';
      gradCtx.fillText(l, x + barW/2 - 3, TOP_PAD + maxBarH + 14);
    }}
  }}
}}

// ---- draw LoRA norms (grouped bars per target) ---------------------------
function drawLoRA(idx) {{
  if (!loraCtx || !HAS_LORA || loraTargets.length === 0) return;
  const rec = RECORDS[idx];
  loraCtx.clearRect(0, 0, loraCanvas.width, loraCanvas.height);
  loraCtx.fillStyle = '#0a0a0f';
  loraCtx.fillRect(0, 0, loraCanvas.width, loraCanvas.height);

  // one row per target, bars per layer
  const barW = (loraCanvas.width - PAD * 2 - LABEL_W) / N_LAYERS - 2;
  const rowH = LORA_ROW_H;
  
  for (let t = 0; t < loraTargets.length; t++) {{
    const target = loraTargets[t];
    const rowY = TOP_PAD + t * (rowH + 4);

    // target label
    loraCtx.fillStyle = '#a855f7';
    loraCtx.font = '500 9px "JetBrains Mono", monospace';
    const label = target.replace('_proj', '');
    loraCtx.fillText(label, 8, rowY + rowH / 2 + 3);

    // bars per layer
    for (let l = 0; l < N_LAYERS; l++) {{
      const key = `${{l}}.${{target}}`;
      const loraVal = rec.lora_norms ? (rec.lora_norms[key] || 0) : 0;
      const normVal = Math.min(1, loraVal / globalLoraMax);
      const x = PAD + LABEL_W + l * (barW + 2);
      const barH = (rowH - 4) * normVal;
      const y = rowY + (rowH - 4) - barH;

      // Glow effect
      if (normVal > 0.5) {{
        loraCtx.shadowColor = '#a855f7';
        loraCtx.shadowBlur = normVal * 6;
      }} else {{
        loraCtx.shadowBlur = 0;
      }}

      loraCtx.fillStyle = loraColor(normVal);
      loraCtx.beginPath();
      loraCtx.roundRect(x, y, barW - 1, barH || 1, 2);
      loraCtx.fill();
      loraCtx.shadowBlur = 0;
    }}
  }}
}}

// ---- combined draw function ----------------------------------------------
function draw(idx) {{
  drawActivations(idx);
  drawGradients(idx);
  drawLoRA(idx);
  const rec = RECORDS[idx];
  
  // Update UI elements
  document.getElementById('step-label').textContent = `Step ${{rec.step}}`;
  document.getElementById('loss-label').textContent = rec.loss != null ? `loss ${{rec.loss.toFixed(4)}}` : '';
  
  // Update progress bar
  const progress = (idx / (RECORDS.length - 1)) * 100;
  sliderProgress.style.width = `${{progress}}%`;
}}

draw(0);

// ---- playback -----------------------------------------------------------
let playing  = false;
let frameMs  = 300;
let timer    = null;
let curIdx   = 0;

function updateSpeedLabel() {{
  const speed = (300 / frameMs).toFixed(1);
  document.getElementById('speed-label').textContent = `${{speed}}×`;
}}

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
  const btn = document.getElementById('btn-play');
  btn.textContent = '⏸';
  btn.classList.add('active');
  timer = setTimeout(tick, frameMs);
}}

function stopPlay() {{
  playing = false;
  clearTimeout(timer);
  const btn = document.getElementById('btn-play');
  btn.textContent = '▶';
  btn.classList.remove('active');
}}

function togglePlay() {{
  if (playing) stopPlay();
  else {{ if (curIdx >= RECORDS.length - 1) setIdx(0); startPlay(); }}
}}

document.getElementById('btn-play').onclick = togglePlay;
slider.oninput = () => {{ stopPlay(); setIdx(+slider.value); }};

document.getElementById('btn-slower').onclick = () => {{ 
  frameMs = Math.min(2000, frameMs + 50); 
  updateSpeedLabel();
}};
document.getElementById('btn-faster').onclick = () => {{ 
  frameMs = Math.max(50, frameMs - 50); 
  updateSpeedLabel();
}};

document.getElementById('btn-diff').onclick = function() {{
  showDiff = !showDiff;
  this.textContent = showDiff ? 'Diff' : 'Absolute';
  this.classList.toggle('active', showDiff);
  draw(curIdx);
}};

// ---- keyboard shortcuts --------------------------------------------------
document.addEventListener('keydown', (e) => {{
  if (e.target.tagName === 'INPUT') return;
  switch(e.key) {{
    case ' ':
      e.preventDefault();
      togglePlay();
      break;
    case 'ArrowLeft':
      e.preventDefault();
      stopPlay();
      setIdx(curIdx - 1);
      break;
    case 'ArrowRight':
      e.preventDefault();
      stopPlay();
      setIdx(curIdx + 1);
      break;
    case 'd':
    case 'D':
      document.getElementById('btn-diff').click();
      break;
  }}
}});

updateSpeedLabel();
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
