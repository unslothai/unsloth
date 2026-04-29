// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { cn } from "@/lib/utils";
import { ArrowExpandDiagonal01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  type ReactElement,
  useCallback,
  useEffect,
  useRef,
  useState,
  useSyncExternalStore,
} from "react";
import { createPortal } from "react-dom";
import { useActivationData } from "@/features/training/hooks/use-activation-data";
import type { ActivationMetadata, ActivationRecord } from "@/features/training/api/train-api";

// ── Color palette ────────────────────────────────────────────────────────────
// Grey (low activity) → Unsloth green (high activity); red-500 for outliers.
// The grey anchor is theme-aware: light grey on dark backgrounds, dark grey on
// light backgrounds. Values chosen for ~7–9:1 contrast against each bg.

const PALETTE_SIZE = 256;
// Dark mode:  slate-700 → Unsloth green  |  outliers: red
// Light mode: white    → red-500         |  outliers: Unsloth green
const GREY_DARK:        [number, number, number] = [51,  65,  85];  // slate-700
const GREY_LIGHT:       [number, number, number] = [255, 255, 255]; // white
const COLOR_HIGH_DARK:  [number, number, number] = [22, 197, 139];  // Unsloth green
const COLOR_HIGH_LIGHT: [number, number, number] = [239,  68,  68]; // red-500
const OUTLIER_COLOR_DARK  = "rgb(239,68,68)";   // red-500
const OUTLIER_COLOR_LIGHT = "rgb(22,197,139)";  // Unsloth green
// Blend mode: outliers rendered at the same colour as the palette high-end
const OUTLIER_BLEND_DARK  = `rgb(${COLOR_HIGH_DARK.join(",")})`;   // green — matches high end
const OUTLIER_BLEND_LIGHT = `rgb(${COLOR_HIGH_LIGHT.join(",")})`;  // red   — matches high end

// Reactively tracks the `dark` class on <html> so canvas + React both update
function subscribe(cb: () => void): () => void {
  const observer = new MutationObserver(cb);
  observer.observe(document.documentElement, { attributeFilter: ["class"] });
  return () => observer.disconnect();
}
function getSnapshot(): boolean {
  return document.documentElement.classList.contains("dark");
}
function useIsDark(): boolean {
  return useSyncExternalStore(subscribe, getSnapshot, () => false);
}

// ── Grid layout constants ─────────────────────────────────────────────────────
const LEFT_MARGIN   = 44;  // px — space for Y-axis neuron-index labels
const BOTTOM_MARGIN = 28;  // px — space for X-axis layer labels + title
const CELL_SIZE = 8;        // px — drawn width/height of each squircle cell
const CELL_GAP  = 1;        // px — gap between cells (makes squircle corners visible)
const CELL_SLOT = CELL_SIZE + CELL_GAP; // slot size used for positioning & hit-testing
const CELL_RADIUS = 2;      // corner radius for squircle look

function buildPalette(
  low:  [number, number, number],
  high: [number, number, number],
): Uint8ClampedArray {
  const palette = new Uint8ClampedArray(PALETTE_SIZE * 4);
  for (let i = 0; i < PALETTE_SIZE; i++) {
    const t = i / (PALETTE_SIZE - 1);
    palette[i * 4 + 0] = Math.round(low[0] + t * (high[0] - low[0]));
    palette[i * 4 + 1] = Math.round(low[1] + t * (high[1] - low[1]));
    palette[i * 4 + 2] = Math.round(low[2] + t * (high[2] - low[2]));
    palette[i * 4 + 3] = 255;
  }
  return palette;
}

const PALETTE_DARK  = buildPalette(GREY_DARK,  COLOR_HIGH_DARK);
const PALETTE_LIGHT = buildPalette(GREY_LIGHT, COLOR_HIGH_LIGHT);
// Tooltip text palettes — inverted low-end so text is readable on the popover bg:
// dark mode:  white  → Unsloth green  |  light mode: dark-grey → red-500
const TOOLTIP_PALETTE_DARK  = buildPalette([255, 255, 255], COLOR_HIGH_DARK);
const TOOLTIP_PALETTE_LIGHT = buildPalette([55,  65,  81],  COLOR_HIGH_LIGHT);

function paletteColor(t: number, palette: Uint8ClampedArray): string {
  const idx = Math.min(PALETTE_SIZE - 1, Math.max(0, Math.round(t * (PALETTE_SIZE - 1))));
  return `rgb(${palette[idx * 4]},${palette[idx * 4 + 1]},${palette[idx * 4 + 2]})`;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

// Rounds a minimum stride up to the nearest "nice" interval so axis ticks land
// on round numbers (e.g. every 5, 10, 16, 32 layers rather than every 7).
function niceInterval(minStride: number): number {
  const steps = [1, 2, 4, 5, 8, 10, 16, 20, 25, 32, 50, 64, 100, 128, 200, 256];
  for (const s of steps) {
    if (s >= minStride) return s;
  }
  return Math.ceil(minStride / 100) * 100;
}

// ── Layout type (returned by drawHeatmap so hover handler can map pixels → cells)

type HeatmapLayout = {
  cellSlot: number;         // slot size (CELL_SIZE + CELL_GAP) for hit-testing (canvas buffer pixels)
  leftMargin: number;       // left axis margin in canvas buffer pixels (for hit-testing)
  numLayers: number;
  numChannels: number;
  layerKeys: string[];
  capturedChannels: number[];
  globalMax: number;        // true maximum (for tooltip display)
  clampMax: number;         // p99 percentile used as color-scale ceiling
  transposed: boolean;      // true when channels→X, layers→Y (expanded landscape mode)
};

// ── Canvas drawing ────────────────────────────────────────────────────────────

function drawHeatmap(
  canvas: HTMLCanvasElement,
  record: ActivationRecord,
  capturedChannels: number[],
  outlierColor: string,
  palette: Uint8ClampedArray,
  transposed = false,
  scale = 1,
): HeatmapLayout | null {
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;

  const layerKeys = Object.keys(record.layers).sort((a, b) => Number(a) - Number(b));
  const numLayers = layerKeys.length;
  if (numLayers === 0) return null;

  const numChannels = capturedChannels.length || record.layers[layerKeys[0]].mean_abs.length;
  if (numChannels === 0) return null;

  // Scale all pixel measurements so expanded mode draws a true HD buffer.
  const s       = scale;
  const cSlot   = CELL_SLOT   * s;  // per-cell slot (cell + gap)
  const cSize   = CELL_SIZE   * s;  // drawn cell size
  const cRadius = CELL_RADIUS * s;  // corner radius
  const lMargin = LEFT_MARGIN   * s;  // left axis area
  const bMargin = BOTTOM_MARGIN * s;  // bottom axis area

  // transposed=false (normal/compact): layers → X, channels → Y  (portrait)
  // transposed=true  (expanded):       channels → X, layers → Y  (landscape)
  const gridW = (transposed ? numChannels : numLayers)  * cSlot;
  const gridH = (transposed ? numLayers  : numChannels) * cSlot;

  // Only resize when dimensions actually change to avoid layout reflow on every draw.
  const newW = Math.round(lMargin + gridW);
  const newH = Math.round(gridH + bMargin);
  if (canvas.width !== newW || canvas.height !== newH) {
    canvas.width  = newW;
    canvas.height = newH;
  }

  // Collect all values; normalise by p99 to avoid outliers washing out the colour scale
  const allVals: number[] = [];
  for (const key of layerKeys) {
    for (const v of record.layers[key].mean_abs) allVals.push(v);
  }
  allVals.sort((a, b) => a - b);
  const globalMax = allVals[allVals.length - 1] || 1;
  const p99Idx = Math.min(Math.floor(0.99 * allVals.length), allVals.length - 1);
  const clampMax = allVals[p99Idx] > 0 ? allVals[p99Idx] : globalMax;

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // ── Draw squircle cells ──
  for (let li = 0; li < numLayers; li++) {
    const layer = record.layers[layerKeys[li]];
    for (let ci = 0; ci < layer.mean_abs.length; ci++) {
      const v  = layer.mean_abs[ci];
      const cx = lMargin + (transposed ? ci : li) * cSlot;
      const cy =           (transposed ? li : ci) * cSlot;
      ctx.fillStyle = v > clampMax ? outlierColor : paletteColor(v / clampMax, palette);
      ctx.beginPath();
      ctx.roundRect(cx, cy, cSize, cSize, cRadius);
      ctx.fill();
    }
  }

  const LABEL_COLOR = "rgba(160,160,160,0.9)";
  const TITLE_COLOR = "rgba(120,120,120,0.75)";

  // ── Y-axis tick labels ──
  // Target ~10 Y labels regardless of buffer scale (cSlot varies with dpr).
  const yCount  = transposed ? numLayers  : numChannels;
  const yStride = niceInterval(Math.max(1, Math.ceil(yCount / 10)));
  ctx.fillStyle     = LABEL_COLOR;
  ctx.font          = `${10 * s}px monospace`;
  ctx.textAlign     = "right";
  ctx.textBaseline  = "middle";
  for (let yi = 0; yi < yCount; yi += yStride) {
    const label = transposed ? String(layerKeys[yi]) : String(capturedChannels[yi] ?? yi);
    ctx.fillText(label, lMargin - 4 * s, yi * cSlot + cSize / 2);
  }
  if ((yCount - 1) % yStride !== 0) {
    const label = transposed
      ? String(layerKeys[yCount - 1])
      : String(capturedChannels[yCount - 1] ?? yCount - 1);
    ctx.fillText(label, lMargin - 4 * s, (yCount - 1) * cSlot + cSize / 2);
  }

  // Y-axis title (rotated)
  ctx.save();
  ctx.translate(8 * s, gridH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillStyle    = TITLE_COLOR;
  ctx.font         = `${11 * s}px sans-serif`;
  ctx.textAlign    = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(transposed ? "Layer" : "Neuron", 0, 0);
  ctx.restore();

  // ── X-axis tick labels ──
  // Target ~12 X labels regardless of buffer scale.
  const xCount  = transposed ? numChannels : numLayers;
  const xStride = niceInterval(Math.max(1, Math.ceil(xCount / 12)));
  ctx.fillStyle     = LABEL_COLOR;
  ctx.font          = `${10 * s}px monospace`;
  ctx.textAlign     = "center";
  ctx.textBaseline  = "top";
  for (let xi = 0; xi < xCount; xi += xStride) {
    const label = transposed ? String(capturedChannels[xi] ?? xi) : String(layerKeys[xi]);
    ctx.fillText(label, lMargin + xi * cSlot + cSize / 2, gridH + 3 * s);
  }
  if ((xCount - 1) % xStride !== 0) {
    const label = transposed
      ? String(capturedChannels[xCount - 1] ?? xCount - 1)
      : String(layerKeys[xCount - 1]);
    ctx.fillText(label, lMargin + (xCount - 1) * cSlot + cSize / 2, gridH + 3 * s);
  }

  // X-axis title
  ctx.fillStyle    = TITLE_COLOR;
  ctx.font         = `${11 * s}px sans-serif`;
  ctx.textAlign    = "center";
  ctx.textBaseline = "bottom";
  ctx.fillText(transposed ? "Channel →" : "Layer →", lMargin + gridW / 2, canvas.height);

  return { cellSlot: cSlot, leftMargin: lMargin, numLayers, numChannels, layerKeys, capturedChannels, globalMax, clampMax, transposed };
}

// ── Types ─────────────────────────────────────────────────────────────────────

type Props = {
  isTraining: boolean;
  jobId: string | null;
};

// ── Tooltip type ─────────────────────────────────────────────────────────────

type TooltipState = {
  clientX: number;
  clientY: number;
  layerKey: string;
  channelIdx: number;
  value: number;      // raw activation
  normalized: number; // 0-1 for palette color
};

// ── Heatmap canvas element ────────────────────────────────────────────────────

function HeatmapCanvas({
  record,
  metadata,
  expanded = false,
  outlierBlend = false,
}: {
  record: ActivationRecord | null;
  metadata: ActivationMetadata | null;
  expanded?: boolean;
  outlierBlend?: boolean;
}): ReactElement {
  const canvasRef        = useRef<HTMLCanvasElement>(null);
  const wrapperRef       = useRef<HTMLDivElement>(null);
  const layoutRef        = useRef<HeatmapLayout | null>(null);
  const imperativeDrawRef = useRef<(() => void) | null>(null);
  const [tooltip, setTooltip] = useState<TooltipState | null>(null);
  const capturedChannels = metadata?.captured_channels ?? [];
  const isDark = useIsDark();
  const palette        = isDark ? PALETTE_DARK         : PALETTE_LIGHT;
  const tooltipPalette = isDark ? TOOLTIP_PALETTE_DARK : TOOLTIP_PALETTE_LIGHT;
  const outlierColor   = outlierBlend
    ? (isDark ? OUTLIER_BLEND_DARK  : OUTLIER_BLEND_LIGHT)
    : (isDark ? OUTLIER_COLOR_DARK  : OUTLIER_COLOR_LIGHT);

  // Keep imperativeDrawRef in sync with latest closure so ResizeObserver can
  // call the current draw function without going through React state.
  useEffect(() => {
    function draw() {
      const canvas = canvasRef.current;
      if (!canvas || !record) return;

      let physicalScale = 1;
      if (expanded) {
        const cw = wrapperRef.current?.offsetWidth ?? 0;
        if (cw > 0) {
          const lKeys   = Object.keys(record.layers);
          const numLays = lKeys.length;
          const numCh   = capturedChannels.length || record.layers[lKeys[0]]?.mean_abs.length || 1;
          // expanded → transposed=true: channels on X axis, layers on Y axis
          const naturalW = LEFT_MARGIN + numCh   * CELL_SLOT;
          const naturalH = numLays    * CELL_SLOT + BOTTOM_MARGIN;
          // Fill as much width as possible; cap height to ~65% of viewport
          const maxH        = window.innerHeight * 0.65;
          const visualScale = Math.min(cw / naturalW, maxH / naturalH, 8);
          const dpr         = window.devicePixelRatio || 1;
          physicalScale     = visualScale * dpr;
          canvas.style.width  = `${Math.round(naturalW * visualScale)}px`;
          canvas.style.height = `${Math.round(naturalH * visualScale)}px`;
        }
      } else {
        canvas.style.width  = "";
        canvas.style.height = "";
      }

      layoutRef.current = drawHeatmap(canvas, record, capturedChannels, outlierColor, palette, expanded, physicalScale);
    }
    imperativeDrawRef.current = draw;
    draw();
  }, [record, capturedChannels, outlierColor, palette, expanded]);

  // Responsive resize: observe wrapper and redraw imperatively — no React
  // state update cycle, so the canvas tracks window resize with zero lag.
  useEffect(() => {
    if (!expanded) return;
    const el = wrapperRef.current;
    if (!el) return;
    // Initial sizing pass after dialog finishes mounting
    imperativeDrawRef.current?.();
    const ro = new ResizeObserver(() => imperativeDrawRef.current?.());
    ro.observe(el);
    return () => ro.disconnect();
  }, [expanded]);

  const handleMouseMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const canvas = e.currentTarget;
      const layout = layoutRef.current;
      if (!layout || !record) return;

      // Map CSS pixels → canvas pixels (handles any CSS scaling)
      const rect   = canvas.getBoundingClientRect();
      const scaleX = canvas.width  / rect.width;
      const scaleY = canvas.height / rect.height;
      const cx = (e.clientX - rect.left) * scaleX;
      const cy = (e.clientY - rect.top)  * scaleY;

      const li = layout.transposed
        ? Math.floor(cy / layout.cellSlot)
        : Math.floor((cx - layout.leftMargin) / layout.cellSlot);
      const ci = layout.transposed
        ? Math.floor((cx - layout.leftMargin) / layout.cellSlot)
        : Math.floor(cy / layout.cellSlot);

      if (li < 0 || li >= layout.numLayers || ci < 0 || ci >= layout.numChannels) {
        setTooltip(null);
        return;
      }

      const layerKey = layout.layerKeys[li];
      const layer    = record.layers[layerKey];
      if (!layer || ci >= layer.mean_abs.length) { setTooltip(null); return; }

      const value      = layer.mean_abs[ci];
      const isOutlier  = value > layout.clampMax;
      const normalized = isOutlier ? -1 : value / layout.clampMax;
      const channelIdx = layout.capturedChannels[ci] ?? ci;

      setTooltip({ clientX: e.clientX, clientY: e.clientY, layerKey, channelIdx, value, normalized });
    },
    [record],
  );

  const handleMouseLeave = useCallback(() => setTooltip(null), []);

  if (!record) {
    return (
      <div className="flex h-[120px] w-full items-center justify-center rounded border border-border/40">
        <p className="text-xs text-muted-foreground">No activation data yet</p>
      </div>
    );
  }

  return (
    <>
      <div ref={wrapperRef} className={expanded ? "w-full overflow-hidden" : "overflow-x-auto"}>
        <canvas
          ref={canvasRef}
          className="block cursor-crosshair"
          style={expanded ? { imageRendering: "auto" } : undefined}
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
        />
      </div>

      {tooltip && createPortal(
        <div
          style={{
            position:      "fixed",
            left:          tooltip.clientX + 14,
            top:           tooltip.clientY - 10,
            pointerEvents: "none",
            zIndex:        9999,
          }}
          className="rounded-lg border border-border/60 bg-popover text-popover-foreground px-2.5 py-1.5 shadow-lg"
        >
          <div className="flex flex-col gap-0.5 text-[11px]">
            <div className="flex items-center justify-between gap-4">
              <span className="text-muted-foreground">Layer</span>
              <span className="font-mono font-medium tabular-nums">{tooltip.layerKey}</span>
            </div>
            <div className="flex items-center justify-between gap-4">
              <span className="text-muted-foreground">Channel</span>
              <span className="font-mono font-medium tabular-nums">{tooltip.channelIdx}</span>
            </div>
            <div className="mt-0.5 border-t border-border/40 pt-1 flex items-center justify-between gap-4">
              <span className="text-muted-foreground">Activation</span>
              <span
                className="font-mono font-semibold tabular-nums"
                style={{ color: tooltip.normalized < 0 ? outlierColor : paletteColor(tooltip.normalized, tooltipPalette) }}
              >
                {tooltip.value.toFixed(4)}{tooltip.normalized < 0 ? " ↑" : ""}
              </span>
            </div>
          </div>
        </div>,
        document.body,
      )}
    </>
  );
}

// ── Color legend bar ──────────────────────────────────────────────────────────

function ColorLegend({
  outlierBlend = false,
  onToggleBlend,
}: {
  outlierBlend?: boolean;
  onToggleBlend?: () => void;
}): ReactElement {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const isDark = useIsDark();
  const palette = isDark ? PALETTE_DARK : PALETTE_LIGHT;
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    canvas.width = canvas.offsetWidth || 200;
    canvas.height = 8;
    const grad = ctx.createLinearGradient(0, 0, canvas.width, 0);
    grad.addColorStop(0, paletteColor(0, palette));
    grad.addColorStop(1, paletteColor(1, palette));
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  }, [isDark, palette]);

  const swatchColor = outlierBlend
    ? (isDark ? OUTLIER_BLEND_DARK  : OUTLIER_BLEND_LIGHT)
    : (isDark ? OUTLIER_COLOR_DARK  : OUTLIER_COLOR_LIGHT);

  return (
    <div className="flex items-center gap-2">
      <span className="text-[10px] text-muted-foreground">Low</span>
      <canvas ref={canvasRef} className="h-2 flex-1 rounded-full" />
      <span className="text-[10px] text-muted-foreground">High</span>
      <span className="mx-1 text-[10px] text-muted-foreground/50">·</span>
      <span
        className="inline-block h-2 w-2 rounded-sm flex-shrink-0"
        style={{ backgroundColor: swatchColor }}
      />
      <Tooltip>
        <TooltipTrigger asChild>
          <button
            type="button"
            onClick={onToggleBlend}
            className={cn(
              "inline-flex cursor-pointer items-center rounded-full border px-1.5 py-0.5 text-[10px] font-medium select-none transition-colors",
              outlierBlend
                ? "border-orange-400/60 bg-orange-400/20 text-orange-400"
                : "border-current/20 bg-muted text-muted-foreground hover:text-foreground",
            )}
          >
            Outlier (&gt;p99)
          </button>
        </TooltipTrigger>
        <TooltipContent className={cn("max-w-[260px]", isDark ? "dark" : "light")}>
          <p className="font-medium mb-1">Outlier detection</p>
          <p className="text-xs/relaxed font-normal">
            All neuron values are sorted and the 99th percentile
            (<code className="font-mono">p99</code>) is used as the colour-scale
            ceiling. Any cell whose activation exceeds <code className="font-mono">p99</code> is
            classified as an outlier and rendered in the accent colour instead
            of the normal gradient.
          </p>
          <p className="text-xs/relaxed font-normal mt-1.5 border-t border-border/40 pt-1.5">
            <span className="font-medium">Click the pill</span> to toggle blend mode — when active
            (orange) outliers use the same colour as the high end of the scale so they blend in,
            making fine gradations easier to spot.
          </p>
        </TooltipContent>
      </Tooltip>
    </div>
  );
}

// ── Replay controls ───────────────────────────────────────────────────────────

const SPEEDS = [0.5, 1, 2, 3, 5, 10] as const;
type Speed = (typeof SPEEDS)[number];

function ReplayControls({
  stepIndex,
  totalSteps,
  onStepChange,
  currentStep,
}: {
  stepIndex: number;
  totalSteps: number;
  onStepChange: (index: number) => void;
  currentStep: number;
}): ReactElement {
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState<Speed>(1);
  const [loop, setLoop] = useState(false);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const stopPlayback = useCallback(() => {
    if (intervalRef.current !== null) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    setPlaying(false);
  }, []);

  const startPlayback = useCallback(() => {
    if (intervalRef.current !== null) clearInterval(intervalRef.current);
    setPlaying(true);
    const ms = Math.round(1000 / speed);
    intervalRef.current = setInterval(() => {
      onStepChange(-1); // signal "advance by 1"
    }, ms);
  }, [speed, onStepChange]);

  // Restart interval when speed changes while playing
  useEffect(() => {
    if (!playing) return;
    startPlayback();
    return stopPlayback;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [speed]);

  // At end of playback: loop or stop
  useEffect(() => {
    if (playing && stepIndex >= totalSteps - 1) {
      if (loop) {
        onStepChange(0);
      } else {
        stopPlayback();
      }
    }
  }, [playing, stepIndex, totalSteps, loop, stopPlayback, onStepChange]);

  // Cleanup on unmount
  useEffect(() => () => stopPlayback(), [stopPlayback]);

  const handleToggle = () => {
    if (playing) {
      stopPlayback();
    } else {
      if (stepIndex >= totalSteps - 1) {
        onStepChange(0); // rewind
      }
      startPlayback();
    }
  };

  if (totalSteps === 0) return <></>;

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center gap-2">
        {/* Play/Pause */}
        <Button
          size="sm"
          variant="outline"
          className="h-6 w-16 text-[11px]"
          onClick={handleToggle}
        >
          {playing ? "⏸ Pause" : "▶ Play"}
        </Button>

        {/* Step slider */}
        <input
          type="range"
          min={0}
          max={Math.max(0, totalSteps - 1)}
          value={stepIndex}
          onChange={(e) => {
            stopPlayback();
            onStepChange(Number(e.target.value));
          }}
          className="h-1 flex-1 cursor-pointer accent-primary"
        />

        {/* Step label */}
        <span className="min-w-[4rem] text-right text-[11px] tabular-nums text-muted-foreground">
          Step {currentStep}
        </span>
      </div>

      {/* Speed selector + loop toggle */}
      <div className="flex items-center gap-1.5">
        <span className="text-[10px] text-muted-foreground">Speed</span>
        {SPEEDS.map((s) => (
          <button
            key={s}
            type="button"
            onClick={() => setSpeed(s)}
            className={cn(
              "rounded px-1.5 py-0.5 text-[10px] font-medium transition-colors",
              speed === s
                ? "bg-primary text-primary-foreground"
                : "text-muted-foreground hover:text-foreground",
            )}
          >
            {s}×
          </button>
        ))}
        {/* Loop toggle */}
        <button
          type="button"
          title={loop ? "Loop on" : "Loop off"}
          aria-pressed={loop}
          onClick={() => setLoop((l) => !l)}
          className={cn(
            "ml-1 rounded p-0.5 text-[12px] leading-none transition-colors",
            loop
              ? "text-primary"
              : "text-muted-foreground hover:text-foreground",
          )}
        >
          {/* repeat/loop glyph */}
          <svg
            viewBox="0 0 20 20"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.8"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="h-3.5 w-3.5"
          >
            <path d="M4 12a6 6 0 1 0 1.5-3.9" />
            <polyline points="1 6 4 8 5.5 5.5" />
          </svg>
        </button>
      </div>
    </div>
  );
}

// ── Main section ──────────────────────────────────────────────────────────────

export function NeuronHeatmapSection({ isTraining }: Props): ReactElement {
  const { metadata, records, loading } = useActivationData({ isTraining });

  const [stepIndex, setStepIndex] = useState<number>(0);

  // During training: always show the latest record
  const displayIndex = isTraining ? Math.max(0, records.length - 1) : stepIndex;
  const record: ActivationRecord | null = records[displayIndex] ?? null;

  // Keep step index at end after training completes
  useEffect(() => {
    if (!isTraining) setStepIndex(Math.max(0, records.length - 1));
  }, [records.length, isTraining]);

  const handleStepChange = useCallback(
    (idx: number) => {
      if (idx === -1) {
        setStepIndex((prev) => Math.min(prev + 1, records.length - 1));
      } else {
        setStepIndex(Math.max(0, Math.min(idx, records.length - 1)));
      }
    },
    [records.length],
  );

  const numLayers = metadata?.num_layers ?? (record ? Object.keys(record.layers).length : 0);
  const numChannels = metadata?.captured_channels?.length ?? 0;
  const [expanded, setExpanded] = useState(false);
  const [outlierBlend, setOutlierBlend] = useState(false);
  const toggleOutlierBlend = useCallback(() => setOutlierBlend((v) => !v), []);

  return (
    <Card size="sm" className="shadow-border border border-border/60 bg-card/90 backdrop-blur-sm">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-1.5">
            <CardTitle className="text-sm ml-1">Neuron Activations</CardTitle>
            {records.length > 0 && (
              <span className="rounded border border-border/60 px-2 py-0.5 text-[10px] leading-tight text-muted-foreground text-center">
                {numLayers} layers ·<br />{numChannels} channels
              </span>
            )}
          </div>
          <div className="flex items-center gap-2">
            <span className="tabular-nums text-[10px] text-muted-foreground flex items-center gap-1">
              {loading && (
                <svg className="animate-spin h-2.5 w-2.5 shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                  <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" strokeLinecap="round"/>
                </svg>
              )}
              {record?.loss != null && <>loss {record.loss.toFixed(3)}</>}
            </span>
            {isTraining && records.length > 0 && (
              <span className="flex items-center gap-1 text-[10px] text-muted-foreground">
                <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-primary" />
                Live
              </span>
            )}
            <button
              type="button"
              onClick={() => setExpanded(true)}
              className="rounded p-1 text-muted-foreground opacity-40 transition-opacity hover:opacity-100 hover:bg-muted/60 hover:text-foreground focus:opacity-100"
              title="Expand chart"
              aria-label="Expand Neuron Activations chart"
            >
              <HugeiconsIcon icon={ArrowExpandDiagonal01Icon} className="size-3.5" />
            </button>
          </div>
        </div>
      </CardHeader>

      <CardContent className="flex flex-col gap-3">
        {/* Skip drawing the card canvas while the dialog is open — avoids a
            duplicate drawHeatmap call on every slider tick which doubles work
            and triggers extra layout reflows. */}
        <HeatmapCanvas record={expanded ? null : record} metadata={metadata} outlierBlend={outlierBlend} />

        {record && <ColorLegend outlierBlend={outlierBlend} onToggleBlend={toggleOutlierBlend} />}

        {!isTraining && records.length > 1 && (
          <ReplayControls
            stepIndex={stepIndex}
            totalSteps={records.length}
            onStepChange={handleStepChange}
            currentStep={record?.step ?? 0}
          />
        )}

        {!loading && records.length === 0 && (
          <p className="text-center text-[11px] text-muted-foreground">
            Activation data will appear here during training
          </p>
        )}
      </CardContent>

      <Dialog open={expanded} onOpenChange={setExpanded}>
        <DialogContent className="w-[90vw] max-w-none sm:max-w-none">
          <DialogHeader>
            <DialogTitle className="font-bold">
              Neuron Activations
              {records.length > 0 && (
                <span className="ml-2 text-sm font-normal text-muted-foreground">
                  {numLayers} layers · {numChannels} channels
                </span>
              )}
            </DialogTitle>
          </DialogHeader>
          <div className="mt-2 w-full flex flex-col gap-3">
            <HeatmapCanvas record={record} metadata={metadata} expanded={true} outlierBlend={outlierBlend} />
            {record && <ColorLegend outlierBlend={outlierBlend} onToggleBlend={toggleOutlierBlend} />}
            {!isTraining && records.length > 1 && (
              <ReplayControls
                stepIndex={stepIndex}
                totalSteps={records.length}
                onStepChange={handleStepChange}
                currentStep={record?.step ?? 0}
              />
            )}
          </div>
        </DialogContent>
      </Dialog>
    </Card>
  );
}
