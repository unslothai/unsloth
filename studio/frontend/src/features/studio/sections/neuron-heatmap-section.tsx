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
import katex from "katex";
import {
  type ReactElement,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  useSyncExternalStore,
} from "react";
import { createPortal } from "react-dom";
import type { ActivationMetadata, ActivationRecord } from "@/features/training/api/train-api";
import {
  computeOverlaySets,
  computeViewValues,
  type OverlaySets,
  type ViewMode,
} from "./activation-stats";
import { InterpretabilityInfoDialog } from "./diagnostics-panel";

// ── Overlay key type (exported for diagnostics-panel) ─────────────────────────
export type OverlayKey = "dead" | "constant" | "onset_dead";

// ── Color palette ─────────────────────────────────────────────────────────────
// Dark mode:  slate-700 → Unsloth green  |  outliers: red-500
// Light mode: lime-400  → Unsloth green (mid) → green-900  |  outliers: red-500

const PALETTE_SIZE = 256;

const GREY_DARK:         [number, number, number] = [51,  65,  85];
const COLOR_HIGH_DARK:   [number, number, number] = [22, 197, 139];

const LIME_LIGHT:        [number, number, number] = [163, 230,  53];
const COLOR_MID_LIGHT:   [number, number, number] = [22,  197, 139];
const COLOR_HIGH_LIGHT:  [number, number, number] = [20,   83,  45];

const OUTLIER_COLOR_DARK  = "rgb(239,68,68)";
const OUTLIER_COLOR_LIGHT = "rgb(239,68,68)";
const OUTLIER_BLEND_DARK  = `rgb(${COLOR_HIGH_DARK.join(",")})`;
const OUTLIER_BLEND_LIGHT = `rgb(${COLOR_HIGH_LIGHT.join(",")})`;

function subscribe(cb: () => void): () => void {
  const observer = new MutationObserver(cb);
  observer.observe(document.documentElement, { attributeFilter: ["class"] });
  return () => observer.disconnect();
}
function useIsDark(): boolean {
  return useSyncExternalStore(
    subscribe,
    () => document.documentElement.classList.contains("dark"),
    () => false,
  );
}

// ── Grid layout constants ─────────────────────────────────────────────────────
const LEFT_MARGIN   = 44;
const BOTTOM_MARGIN = 28;
const CELL_SIZE = 8;
const CELL_GAP  = 1;
const CELL_SLOT = CELL_SIZE + CELL_GAP;
const CELL_RADIUS = 2;
const TOP_MARGIN  = 8;

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

function buildPalette3(
  low:  [number, number, number],
  mid:  [number, number, number],
  high: [number, number, number],
): Uint8ClampedArray {
  const palette = new Uint8ClampedArray(PALETTE_SIZE * 4);
  for (let i = 0; i < PALETTE_SIZE; i++) {
    const t = i / (PALETTE_SIZE - 1);
    let r: number, g: number, b: number;
    if (t <= 0.5) {
      const u = t / 0.5;
      r = Math.round(low[0] + u * (mid[0] - low[0]));
      g = Math.round(low[1] + u * (mid[1] - low[1]));
      b = Math.round(low[2] + u * (mid[2] - low[2]));
    } else {
      const u = (t - 0.5) / 0.5;
      r = Math.round(mid[0] + u * (high[0] - mid[0]));
      g = Math.round(mid[1] + u * (high[1] - mid[1]));
      b = Math.round(mid[2] + u * (high[2] - mid[2]));
    }
    palette[i * 4 + 0] = r;
    palette[i * 4 + 1] = g;
    palette[i * 4 + 2] = b;
    palette[i * 4 + 3] = 255;
  }
  return palette;
}

const PALETTE_DARK  = buildPalette(GREY_DARK, COLOR_HIGH_DARK);
const PALETTE_LIGHT = buildPalette3(LIME_LIGHT, COLOR_MID_LIGHT, COLOR_HIGH_LIGHT);
const TOOLTIP_PALETTE_DARK  = buildPalette([255, 255, 255], COLOR_HIGH_DARK);
const TOOLTIP_PALETTE_LIGHT = buildPalette3([55, 65, 81], COLOR_MID_LIGHT, COLOR_HIGH_LIGHT);

function paletteColor(t: number, palette: Uint8ClampedArray): string {
  const idx = Math.min(PALETTE_SIZE - 1, Math.max(0, Math.round(t * (PALETTE_SIZE - 1))));
  return `rgb(${palette[idx * 4]},${palette[idx * 4 + 1]},${palette[idx * 4 + 2]})`;
}

function niceInterval(minStride: number): number {
  const steps = [1, 2, 4, 5, 8, 10, 16, 20, 25, 32, 50, 64, 100, 128, 200, 256];
  for (const s of steps) {
    if (s >= minStride) return s;
  }
  return Math.ceil(minStride / 100) * 100;
}

// ── Layout type ───────────────────────────────────────────────────────────────

type HeatmapLayout = {
  cellSlot: number;
  leftMargin: number;
  topMargin: number;
  numLayers: number;
  numChannels: number;
  layerKeys: string[];
  capturedChannels: number[];
  globalMax: number;
  clampMax: number;
  transposed: boolean;
};

// ── Overlay canvas colors ─────────────────────────────────────────────────────

const OVERLAY_DEFS: Array<[OverlayKey, string]> = [
  ["dead",       "rgba(96,165,250,0.55)"],
  ["constant",   "rgba(251,146,60,0.5)"],
  ["onset_dead", "rgba(147,51,234,0.65)"],
];

// ── Canvas drawing ────────────────────────────────────────────────────────────

function drawHeatmap(
  canvas: HTMLCanvasElement,
  values: Record<string, number[]>,
  capturedChannels: number[],
  outlierColor: string,
  palette: Uint8ClampedArray,
  activeOverlays: Set<OverlayKey>,
  overlaySets: OverlaySets,
  transposed = false,
  scale = 1,
): HeatmapLayout | null {
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;

  const layerKeys = Object.keys(values).sort((a, b) => Number(a) - Number(b));
  const numLayers = layerKeys.length;
  if (numLayers === 0) return null;

  const numChannels = capturedChannels.length || values[layerKeys[0]]?.length || 0;
  if (numChannels === 0) return null;

  const s       = scale;
  const cSlot   = CELL_SLOT   * s;
  const cSize   = CELL_SIZE   * s;
  const cRadius = CELL_RADIUS * s;
  const lMargin = LEFT_MARGIN   * s;
  const bMargin = BOTTOM_MARGIN * s;
  const tMargin = TOP_MARGIN    * s;

  const gridW = (transposed ? numChannels : numLayers)  * cSlot;
  const gridH = (transposed ? numLayers  : numChannels) * cSlot;

  const newW = Math.round(lMargin + gridW);
  const newH = Math.round(tMargin + gridH + bMargin);
  if (canvas.width !== newW || canvas.height !== newH) {
    canvas.width  = newW;
    canvas.height = newH;
  }

  // Collect all values; normalise by p99
  const allVals: number[] = [];
  for (const key of layerKeys) {
    for (const v of (values[key] ?? [])) allVals.push(v);
  }
  allVals.sort((a, b) => a - b);
  const globalMax = allVals[allVals.length - 1] || 1;
  const p99Idx = Math.min(Math.floor(0.99 * allVals.length), allVals.length - 1);
  const clampMax = allVals[p99Idx] > 0 ? allVals[p99Idx] : globalMax;

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // ── Draw squircle cells ──
  for (let li = 0; li < numLayers; li++) {
    const layerVals = values[layerKeys[li]] ?? [];
    for (let ci = 0; ci < layerVals.length; ci++) {
      const v  = layerVals[ci];
      const cx = lMargin + (transposed ? ci : li) * cSlot;
      const cy = tMargin + (transposed ? li : ci) * cSlot;
      ctx.fillStyle = v > clampMax ? outlierColor : paletteColor(v / clampMax, palette);
      ctx.beginPath();
      ctx.roundRect(cx, cy, cSize, cSize, cRadius);
      ctx.fill();
    }
  }

  // ── Overlay pass ──
  if (activeOverlays.size > 0) {
    const layerIndexMap = new Map<string, number>();
    layerKeys.forEach((k, i) => layerIndexMap.set(k, i));

    for (const [overlayKey, overlayColor] of OVERLAY_DEFS) {
      if (!activeOverlays.has(overlayKey)) continue;
      const cellSet =
        overlayKey === "dead"       ? overlaySets.dead :
        overlayKey === "constant"   ? overlaySets.constant :
                                      overlaySets.onsetDead;

      ctx.fillStyle = overlayColor;
      for (const id of cellSet) {
        const colonIdx = id.indexOf(":");
        const layerKey = id.slice(0, colonIdx);
        const ci = Number(id.slice(colonIdx + 1));
        const li = layerIndexMap.get(layerKey) ?? -1;
        if (li < 0 || ci >= numChannels) continue;
        const cx = lMargin + (transposed ? ci : li) * cSlot;
        const cy = tMargin + (transposed ? li : ci) * cSlot;
        ctx.beginPath();
        ctx.roundRect(cx, cy, cSize, cSize, cRadius);
        ctx.fill();
      }
    }
  }

  const LABEL_COLOR = "rgba(160,160,160,0.9)";
  const TITLE_COLOR = "rgba(120,120,120,0.75)";

  // ── Y-axis tick labels ──
  const yCount  = transposed ? numLayers  : numChannels;
  const yStride = niceInterval(Math.max(1, Math.ceil(yCount / 10)));
  ctx.fillStyle     = LABEL_COLOR;
  ctx.font          = `${10 * s}px monospace`;
  ctx.textAlign     = "right";
  ctx.textBaseline  = "middle";
  for (let yi = 0; yi < yCount; yi += yStride) {
    const label = transposed ? String(layerKeys[yi]) : String(capturedChannels[yi] ?? yi);
    ctx.fillText(label, lMargin - 4 * s, tMargin + yi * cSlot + cSize / 2);
  }
  if ((yCount - 1) % yStride !== 0) {
    const label = transposed
      ? String(layerKeys[yCount - 1])
      : String(capturedChannels[yCount - 1] ?? yCount - 1);
    ctx.fillText(label, lMargin - 4 * s, tMargin + (yCount - 1) * cSlot + cSize / 2);
  }

  ctx.save();
  ctx.translate(8 * s, tMargin + gridH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillStyle    = TITLE_COLOR;
  ctx.font         = `${11 * s}px sans-serif`;
  ctx.textAlign    = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(transposed ? "Layer" : "Neuron", 0, 0);
  ctx.restore();

  // ── X-axis tick labels ──
  const xCount  = transposed ? numChannels : numLayers;
  const xStride = niceInterval(Math.max(1, Math.ceil(xCount / 12)));
  ctx.fillStyle     = LABEL_COLOR;
  ctx.font          = `${10 * s}px monospace`;
  ctx.textAlign     = "center";
  ctx.textBaseline  = "top";
  for (let xi = 0; xi < xCount; xi += xStride) {
    const label = transposed ? String(capturedChannels[xi] ?? xi) : String(layerKeys[xi]);
    ctx.fillText(label, lMargin + xi * cSlot + cSize / 2, tMargin + gridH + 3 * s);
  }
  if ((xCount - 1) % xStride !== 0) {
    const label = transposed
      ? String(capturedChannels[xCount - 1] ?? xCount - 1)
      : String(layerKeys[xCount - 1]);
    ctx.fillText(label, lMargin + (xCount - 1) * cSlot + cSize / 2, tMargin + gridH + 3 * s);
  }

  ctx.fillStyle    = TITLE_COLOR;
  ctx.font         = `${11 * s}px sans-serif`;
  ctx.textAlign    = "center";
  ctx.textBaseline = "bottom";
  ctx.fillText(transposed ? "Channel →" : "Layer →", lMargin + gridW / 2, canvas.height);

  return { cellSlot: cSlot, leftMargin: lMargin, topMargin: tMargin, numLayers, numChannels, layerKeys, capturedChannels, globalMax, clampMax, transposed };
}

// ── Types ─────────────────────────────────────────────────────────────────────

type SectionProps = {
  isTraining: boolean;
  records: ActivationRecord[];
  metadata: ActivationMetadata | null;
  loading: boolean;
  record: ActivationRecord | null;
  stepIndex: number;
  onStepChange: (idx: number) => void;
};

type TooltipState = {
  clientX: number;
  clientY: number;
  layerKey: string;
  channelIdx: number;
  value: number;
  normalized: number;
};

// ── KaTeX helpers ─────────────────────────────────────────────────────────────

function KatexDisplay({ latex }: { latex: string }): ReactElement {
  return (
    <span
      className="block overflow-x-auto text-[11px]"
      dangerouslySetInnerHTML={{
        __html: katex.renderToString(latex, { throwOnError: false, displayMode: true }),
      }}
    />
  );
}

// ── Mode pill configs ─────────────────────────────────────────────────────────

interface ViewModeConfig {
  key: ViewMode;
  label: string;
  latex: string;
  description: string;
  requiresGradients?: boolean;
  requiresLoraNorms?: boolean;
  requiresMultipleSteps?: boolean;
}

const VIEW_MODES: ViewModeConfig[] = [
  {
    key: "activations",
    label: "Activations",
    latex: "\\bar{a} \\equiv \\frac{1}{C}\\sum_{i=1}^{C}|x_i|",
    description: "C = sampled channels, xᵢ = activation of channel i. How strongly each neuron fires.",
  },
  {
    key: "gradients",
    label: "Gradients",
    latex: "g \\equiv \\|\\nabla_{W}L\\|_2 = \\sqrt{\\sum_i \\left(\\tfrac{\\partial L}{\\partial w_i}\\right)^2}",
    description: "Per-layer gradient norm. Near zero → vanishing. Very large → exploding.",
    requiresGradients: true,
  },
  {
    key: "lora_norms",
    label: "LoRA",
    latex: "\\|\\Delta W\\|_F \\equiv \\sqrt{\\sum_{i,j}(BA)_{ij}^2}",
    description: "B, A = LoRA matrices. Low norm → adapter barely adapting this layer.",
    requiresLoraNorms: true,
  },
  {
    key: "delta",
    label: "Delta",
    latex: "\\Delta \\equiv |\\bar{a}_t - \\bar{a}_{t-1}|",
    description: "Absolute change from the previous step. High → actively changing right now.",
    requiresMultipleSteps: true,
  },
  {
    key: "trend",
    label: "Trend",
    latex: "\\beta \\equiv \\frac{\\sum(t_i-\\bar{t})(x_i-\\bar{x})}{\\sum(t_i-\\bar{t})^2}",
    description: "|β| shown — slope of mean_abs over training. Large → rapidly changing channel.",
    requiresMultipleSteps: true,
  },
];

interface OverlayConfig {
  key: OverlayKey;
  label: string;
  color: string;
  latex: string;
  description: string;
}

const OVERLAY_CONFIGS: OverlayConfig[] = [
  {
    key: "dead",
    label: "Dead",
    color: "rgb(96,165,250)",
    latex: "\\max_t(\\bar{a}_t) < \\varepsilon,\\quad \\varepsilon = 0.01",
    description: "Channel never exceeded ε — contributes nothing to model output.",
  },
  {
    key: "constant",
    label: "Constant",
    color: "rgb(251,146,60)",
    latex: "CV \\equiv \\sigma/\\mu < 0.05",
    description: "Fires but barely changes over training. Limits expressive capacity.",
  },
  {
    key: "onset_dead",
    label: "Onset Dead",
    color: "rgb(147,51,234)",
    latex: "\\bar{a}_0 \\geq \\varepsilon \\;\\wedge\\; \\max_{t>0}(\\bar{a}_t) < \\varepsilon",
    description: "Was alive at step 0, died mid-training — the critical red flag.",
  },
];

// ── ModePill ──────────────────────────────────────────────────────────────────

interface ModePillProps {
  label: string;
  active: boolean;
  disabled?: boolean;
  disabledReason?: string;
  onClick: () => void;
  latex: string;
  description: string;
  indicatorColor?: string;
  activeColor?: string;
}

function ModePill({
  label,
  active,
  disabled,
  disabledReason,
  onClick,
  latex,
  description,
  indicatorColor,
  activeColor,
}: ModePillProps): ReactElement {
  const activeBorderColor = activeColor ? `${activeColor.replace("rgb", "rgba").replace(")", ", 0.5)")}` : undefined;
  const activeBgColor = activeColor ? `${activeColor.replace("rgb", "rgba").replace(")", ", 0.15)")}` : undefined;

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          disabled={disabled}
          onClick={disabled ? undefined : onClick}
          style={active && activeColor ? { borderColor: activeBorderColor, backgroundColor: activeBgColor, color: activeColor } : undefined}
          className={cn(
            "inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[10px] font-medium transition-colors select-none",
            active && !activeColor
              ? "border-primary/60 bg-primary/15 text-primary"
              : disabled
                ? "border-border/30 text-muted-foreground/40 cursor-not-allowed"
                : !active
                  ? "border-border/50 text-muted-foreground hover:text-foreground hover:border-border cursor-pointer"
                  : "cursor-pointer",
          )}
        >
          {indicatorColor && (
            <span
              className="inline-block h-1.5 w-1.5 rounded-full shrink-0"
              style={{ backgroundColor: active ? indicatorColor : "currentColor" }}
            />
          )}
          {label}
        </button>
      </TooltipTrigger>
      <TooltipContent side="bottom" className="max-w-[300px] p-3">
        {disabled && disabledReason ? (
          <p className="text-xs text-muted-foreground">{disabledReason}</p>
        ) : (
          <div className="flex flex-col gap-1.5">
            <KatexDisplay latex={latex} />
            <p className="text-xs text-muted-foreground leading-relaxed">{description}</p>
          </div>
        )}
      </TooltipContent>
    </Tooltip>
  );
}

// ── HeatmapCanvas ─────────────────────────────────────────────────────────────

function HeatmapCanvas({
  values,
  capturedChannels,
  activeOverlays,
  overlaySets,
  expanded = false,
  outlierBlend = false,
}: {
  values: Record<string, number[]> | null;
  capturedChannels: number[];
  activeOverlays: Set<OverlayKey>;
  overlaySets: OverlaySets;
  expanded?: boolean;
  outlierBlend?: boolean;
}): ReactElement {
  const canvasRef         = useRef<HTMLCanvasElement>(null);
  const wrapperRef        = useRef<HTMLDivElement>(null);
  const layoutRef         = useRef<HeatmapLayout | null>(null);
  const imperativeDrawRef = useRef<(() => void) | null>(null);
  const [tooltip, setTooltip] = useState<TooltipState | null>(null);
  const isDark = useIsDark();
  const palette        = isDark ? PALETTE_DARK         : PALETTE_LIGHT;
  const tooltipPalette = isDark ? TOOLTIP_PALETTE_DARK : TOOLTIP_PALETTE_LIGHT;
  const outlierColor   = outlierBlend
    ? (isDark ? OUTLIER_BLEND_DARK  : OUTLIER_BLEND_LIGHT)
    : (isDark ? OUTLIER_COLOR_DARK  : OUTLIER_COLOR_LIGHT);

  useEffect(() => {
    function draw() {
      const canvas = canvasRef.current;
      if (!canvas || !values) return;

      let physicalScale = 1;
      if (expanded) {
        const cw = wrapperRef.current?.offsetWidth ?? 0;
        if (cw > 0) {
          const lKeys   = Object.keys(values);
          const numLays = lKeys.length;
          const numCh   = capturedChannels.length || values[lKeys[0]]?.length || 1;
          const naturalW = LEFT_MARGIN + numCh   * CELL_SLOT;
          const naturalH = numLays    * CELL_SLOT + BOTTOM_MARGIN + TOP_MARGIN;
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

      layoutRef.current = drawHeatmap(
        canvas, values, capturedChannels, outlierColor, palette,
        activeOverlays, overlaySets, expanded, physicalScale,
      );
    }
    imperativeDrawRef.current = draw;
    draw();
  }, [values, capturedChannels, outlierColor, palette, activeOverlays, overlaySets, expanded]);

  useEffect(() => {
    if (!expanded) return;
    const el = wrapperRef.current;
    if (!el) return;
    imperativeDrawRef.current?.();
    const ro = new ResizeObserver(() => imperativeDrawRef.current?.());
    ro.observe(el);
    return () => ro.disconnect();
  }, [expanded]);

  const handleMouseMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const canvas = e.currentTarget;
      const layout = layoutRef.current;
      if (!layout || !values) return;

      const rect   = canvas.getBoundingClientRect();
      const scaleX = canvas.width  / rect.width;
      const scaleY = canvas.height / rect.height;
      const cx = (e.clientX - rect.left) * scaleX;
      const cy = (e.clientY - rect.top)  * scaleY - layout.topMargin;

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

      const layerKey   = layout.layerKeys[li];
      const layerVals  = values[layerKey];
      if (!layerVals || ci >= layerVals.length) { setTooltip(null); return; }

      const value      = layerVals[ci];
      const isOutlier  = value > layout.clampMax;
      const normalized = isOutlier ? -1 : value / layout.clampMax;
      const channelIdx = layout.capturedChannels[ci] ?? ci;

      setTooltip({ clientX: e.clientX, clientY: e.clientY, layerKey, channelIdx, value, normalized });
    },
    [values],
  );

  const handleMouseLeave = useCallback(() => setTooltip(null), []);

  if (!values) {
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
              <span className="text-muted-foreground">Value</span>
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

// ── ColorLegend ───────────────────────────────────────────────────────────────

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
        <TooltipContent className="max-w-[260px]">
          <p className="font-medium mb-1">Outlier detection</p>
          <p className="text-xs/relaxed font-normal">
            All neuron values are sorted and the 99th percentile
            (<code className="font-mono">p99</code>) is used as the colour-scale
            ceiling. Any cell whose activation exceeds <code className="font-mono">p99</code> is
            classified as an outlier and rendered in the accent colour instead
            of the normal gradient.
          </p>
          <p className="text-xs/relaxed font-normal mt-1.5 border-t border-border/40 pt-1.5">
            <span className="font-medium">Click the pill</span> to toggle blend mode — when active
            (orange) outliers use the same colour as the high end of the scale so they blend in,
            making fine gradations easier to spot.
          </p>
        </TooltipContent>
      </Tooltip>
    </div>
  );
}

// ── ReplayControls ────────────────────────────────────────────────────────────

const SPEEDS = [0.5, 1, 2, 3, 5, 10] as const;
type Speed = (typeof SPEEDS)[number];

export function ReplayControls({
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
      onStepChange(-1);
    }, ms);
  }, [speed, onStepChange]);

  useEffect(() => {
    if (!playing) return;
    startPlayback();
    return stopPlayback;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [speed]);

  useEffect(() => {
    if (playing && stepIndex >= totalSteps - 1) {
      if (loop) {
        onStepChange(0);
      } else {
        stopPlayback();
      }
    }
  }, [playing, stepIndex, totalSteps, loop, stopPlayback, onStepChange]);

  useEffect(() => () => stopPlayback(), [stopPlayback]);

  const handleToggle = () => {
    if (playing) {
      stopPlayback();
    } else {
      if (stepIndex >= totalSteps - 1) onStepChange(0);
      startPlayback();
    }
  };

  if (totalSteps === 0) return <></>;

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center gap-2">
        <Button size="sm" variant="outline" className="h-6 w-16 text-[11px]" onClick={handleToggle}>
          {playing ? "⏸ Pause" : "▶ Play"}
        </Button>
        <input
          type="range"
          min={0}
          max={Math.max(0, totalSteps - 1)}
          value={stepIndex}
          onChange={(e) => { stopPlayback(); onStepChange(Number(e.target.value)); }}
          className="h-1 flex-1 cursor-pointer accent-primary"
        />
        <span className="min-w-[4rem] text-right text-[11px] tabular-nums text-muted-foreground">
          Step {currentStep}
        </span>
      </div>
      <div className="flex items-center gap-1.5">
        <span className="text-[10px] text-muted-foreground">Speed</span>
        {SPEEDS.map((s) => (
          <button
            key={s}
            type="button"
            onClick={() => setSpeed(s)}
            className={cn(
              "rounded px-1.5 py-0.5 text-[10px] font-medium transition-colors",
              speed === s ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:text-foreground",
            )}
          >
            {s}×
          </button>
        ))}
        <button
          type="button"
          title={loop ? "Loop on" : "Loop off"}
          aria-pressed={loop}
          onClick={() => setLoop((l) => !l)}
          className={cn(
            "ml-1 rounded p-0.5 text-[12px] leading-none transition-colors",
            loop ? "text-primary" : "text-muted-foreground hover:text-foreground",
          )}
        >
          <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" className="h-3.5 w-3.5">
            <path d="M4 12a6 6 0 1 0 1.5-3.9" />
            <polyline points="1 6 4 8 5.5 5.5" />
          </svg>
        </button>
      </div>
    </div>
  );
}

// ── NeuronHeatmapSection ──────────────────────────────────────────────────────

export function NeuronHeatmapSection({
  isTraining,
  records,
  metadata,
  loading,
  record,
  stepIndex,
  onStepChange,
}: SectionProps): ReactElement {
  const numLayers   = metadata?.num_layers ?? (record ? Object.keys(record.layers).length : 0);
  const numChannels = metadata?.captured_channels?.length ?? 0;
  const capturedChannels = metadata?.captured_channels ?? [];

  const [expanded,     setExpanded]     = useState(false);
  const [outlierBlend, setOutlierBlend] = useState(false);
  const [viewMode,     setViewMode]     = useState<ViewMode>("activations");
  const [activeOverlays, setActiveOverlays] = useState<Set<OverlayKey>>(new Set());
  const [infoOpen,     setInfoOpen]     = useState(false);

  const toggleOutlierBlend = useCallback(() => setOutlierBlend((v) => !v), []);
  const toggleOverlay = useCallback((key: OverlayKey) => {
    setActiveOverlays((prev) => {
      const next = new Set(prev);
      next.has(key) ? next.delete(key) : next.add(key);
      return next;
    });
  }, []);

  // Compute heatmap values for the current view mode + step
  const computedValues = useMemo(
    () => (record ? computeViewValues(records, stepIndex, viewMode) : null),
    [records, stepIndex, viewMode, record],
  );

  // Compute overlay cell sets (cumulative up to current step)
  const overlaySets = useMemo(
    () => computeOverlaySets(records, stepIndex),
    [records, stepIndex],
  );

  // Disabled reasons for view mode pills
  const hasGradients  = metadata?.capture_gradients ?? false;
  const hasLoraNorms  = metadata?.capture_lora_norms ?? false;
  const hasMultiSteps = records.length >= 2;

  function getDisabledReason(mode: ViewModeConfig): string | undefined {
    if (mode.requiresGradients && !hasGradients)
      return "Enable capture_gradients in activation capture settings to use this view.";
    if (mode.requiresLoraNorms && !hasLoraNorms)
      return "Enable capture_lora_norms in activation capture settings to use this view.";
    if (mode.requiresMultipleSteps && !hasMultiSteps)
      return "Need at least 2 captured steps to use this view.";
    return undefined;
  }

  return (
    <Card size="sm" className="shadow-border border border-border/60 bg-card/90 backdrop-blur-sm">
      <CardHeader className="pb-2">
        {/* Row 1: title + meta + controls */}
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
              onClick={() => setInfoOpen(true)}
              className="rounded p-1 text-muted-foreground opacity-40 transition-opacity hover:opacity-100 hover:bg-muted/60 hover:text-foreground focus:opacity-100 text-[11px] font-medium leading-none"
              title="How to read this chart"
              aria-label="Open interpretability guide"
            >
              ?
            </button>
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

        {/* Row 2: View mode pills */}
        <div className="flex flex-wrap gap-1 mt-2">
          {VIEW_MODES.map((mode) => {
            const disabledReason = getDisabledReason(mode);
            return (
              <ModePill
                key={mode.key}
                label={mode.label}
                active={viewMode === mode.key}
                disabled={!!disabledReason}
                disabledReason={disabledReason}
                onClick={() => setViewMode(mode.key)}
                latex={mode.latex}
                description={mode.description}
              />
            );
          })}
        </div>

        {/* Row 3: Overlay pills */}
        <div className="flex flex-wrap gap-1 mt-1">
          {OVERLAY_CONFIGS.map((ov) => (
            <ModePill
              key={ov.key}
              label={ov.label}
              active={activeOverlays.has(ov.key)}
              onClick={() => toggleOverlay(ov.key)}
              latex={ov.latex}
              description={ov.description}
              indicatorColor={ov.color}
              activeColor={ov.color}
            />
          ))}
        </div>
      </CardHeader>

      <CardContent className="flex flex-col gap-3">
        <HeatmapCanvas
          values={expanded ? null : computedValues}
          capturedChannels={capturedChannels}
          activeOverlays={activeOverlays}
          overlaySets={overlaySets}
          outlierBlend={outlierBlend}
        />

        {record && <ColorLegend outlierBlend={outlierBlend} onToggleBlend={toggleOutlierBlend} />}

        {!loading && records.length === 0 && (
          <p className="text-center text-[11px] text-muted-foreground">
            Activation data will appear here during training
          </p>
        )}
      </CardContent>

      {/* Expanded dialog */}
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
            <HeatmapCanvas
              values={computedValues}
              capturedChannels={capturedChannels}
              activeOverlays={activeOverlays}
              overlaySets={overlaySets}
              expanded={true}
              outlierBlend={outlierBlend}
            />
            {record && <ColorLegend outlierBlend={outlierBlend} onToggleBlend={toggleOutlierBlend} />}
            {!isTraining && records.length > 1 && (
              <ReplayControls
                stepIndex={stepIndex}
                totalSteps={records.length}
                onStepChange={onStepChange}
                currentStep={record?.step ?? 0}
              />
            )}
          </div>
        </DialogContent>
      </Dialog>

      {/* Info dialog */}
      <InterpretabilityInfoDialog open={infoOpen} onOpenChange={setInfoOpen} />
    </Card>
  );
}
