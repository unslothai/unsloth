// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The card every Benchmarks chart is drawn on: background, title, fine print and the hover
// tooltip. The SVG carries literal colours, so Download SVG / PNG saves what is on screen.

import { useTheme } from "@/features/settings";
import { cn } from "@/lib/utils";
import type { MouseEvent, ReactElement, ReactNode, RefObject } from "react";
import { FONT, FOOT_LINE, W, wrapFooter } from "./chart-math";

let probe: CanvasRenderingContext2D | null | undefined;

/** Any CSS colour folded to #rrggbb, since many SVG viewers can't paint oklch(). */
function toPortableColor(value: string): string {
  if (typeof document === "undefined") return value;
  if (probe === undefined) {
    const canvas = document.createElement("canvas");
    canvas.width = 1;
    canvas.height = 1;
    probe = canvas.getContext("2d", { willReadFrequently: true });
  }
  if (!probe) return value;
  probe.clearRect(0, 0, 1, 1);
  probe.fillStyle = "#000";
  probe.fillStyle = value;
  probe.fillRect(0, 0, 1, 1);
  const [r, g, b, a] = probe.getImageData(0, 0, 1, 1).data;
  const hex = (n: number) => n.toString(16).padStart(2, "0");
  return a === 255
    ? `#${hex(r)}${hex(g)}${hex(b)}`
    : `rgba(${r}, ${g}, ${b}, ${(a / 255).toFixed(3)})`;
}

export interface Ink {
  ink: string;
  muted: string;
  grid: string;
  card: string;
}

export function useInk(): Ink {
  // Subscribing re-renders on a theme flip, which is when the tokens change.
  useTheme();
  const style = getComputedStyle(document.documentElement);
  const read = (n: string, fallback: string) =>
    toPortableColor(style.getPropertyValue(n).trim() || fallback);
  return {
    ink: read("--foreground", "#111"),
    muted: read("--muted-foreground", "#777"),
    grid: read("--border", "#ddd"),
    card: read("--card", "#fff"),
  };
}

export interface Tip {
  x: number;
  y: number;
  lines: string[];
}

/** Space above the plot: the title and subtitle, or a sliver when bare. */
export function chartHead(bare: boolean, subtitle?: string): number {
  return bare ? 8 : subtitle ? 70 : 48;
}

export function chartHeight(
  head: number,
  plotH: number,
  footLines: string[],
): number {
  return (
    head + plotH + (footLines.length ? 10 + footLines.length * FOOT_LINE : 0) + 10
  );
}

export function footerLines(bare: boolean, footer: string[]): string[] {
  return bare ? [] : wrapFooter(footer);
}

export function ChartFrame({
  title,
  subtitle,
  footLines,
  bare,
  head,
  plotH,
  height,
  card,
  ink,
  muted,
  svgRef,
  tip,
  onMouseMove,
  onMouseLeave,
  children,
}: {
  title: string;
  subtitle?: string;
  footLines: string[];
  bare: boolean;
  head: number;
  plotH: number;
  height: number;
  card: string;
  ink: string;
  muted: string;
  svgRef?: RefObject<SVGSVGElement | null>;
  tip: Tip | null;
  onMouseMove?: (e: MouseEvent<SVGSVGElement>) => void;
  onMouseLeave: () => void;
  children: ReactNode;
}): ReactElement {
  return (
    <div className="relative mx-auto max-w-[calc(1000px*var(--ui-space-scale,1))]">
      <svg
        ref={svgRef}
        viewBox={`0 0 ${W} ${height}`}
        width="100%"
        className="mx-auto block max-w-[calc(1000px*var(--ui-space-scale,1))]"
        role="img"
        aria-label={title}
        style={{ fontFamily: FONT }}
        onMouseMove={onMouseMove}
        onMouseLeave={onMouseLeave}
      >
        <rect x={0} y={0} width={W} height={height} fill={card} />
        {!bare && (
          <text x={24} y={30} fontSize={18} fontWeight={650} fill={ink}>
            {title}
          </text>
        )}
        {!bare && subtitle && (
          <text x={24} y={52} fontSize={12.5} fill={muted}>
            {subtitle}
          </text>
        )}
        <g transform={`translate(0, ${head})`}>{children}</g>
        {footLines.map((line, i) => (
          <text
            key={`${i}-${line}`}
            x={24}
            y={head + plotH + 14 + i * FOOT_LINE}
            fontSize={11}
            fill={muted}
          >
            {line}
          </text>
        ))}
      </svg>
      {tip && (
        <div
          className="pointer-events-none absolute z-10 min-w-44 rounded-lg border border-border/60 bg-popover px-3 py-2 text-ui-12 shadow-lg"
          style={{
            left: `${(Math.min(tip.x, W - 300) / W) * 100}%`,
            top: `${(tip.y / height) * 100}%`,
            transform: "translate(14px, 10px)",
          }}
        >
          {tip.lines.map((l, i) => (
            <p
              key={l}
              className={cn(
                i === 0
                  ? "font-semibold text-foreground"
                  : "tabular-nums text-muted-foreground",
              )}
            >
              {l}
            </p>
          ))}
        </div>
      )}
    </div>
  );
}
