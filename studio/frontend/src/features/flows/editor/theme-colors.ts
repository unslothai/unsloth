// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const token = (name: string) => `var(--${name})`;
const blend = (a: string, percent: number, b: string) =>
  `color-mix(in oklch, var(--${a}) ${percent}%, var(--${b}))`;

const VIOLET = blend("chart-3", 30, "chart-4");
const INDIGO = blend("chart-3", 60, "chart-4");
const ORANGE = blend("chart-5", 50, "destructive");
const ROSE = blend("chart-4", 50, "destructive");
const TEAL = blend("chart-3", 50, "primary");
const LIME = blend("chart-2", 50, "chart-5");
const LIGHT = blend("foreground", 55, "muted-foreground");

// Catalog hexes (specs.py PORT_COLORS and CATEGORIES) -> Studio's chart, primary and
// destructive tokens, so wires follow the active palette in both themes.
const STUDIO_INK: Record<string, string> = {
  "#a78bfa": VIOLET, // text
  "#c084fc": VIOLET, // story
  "#fbbf24": token("chart-5"), // number
  "#facc15": token("chart-5"), // chart
  "#f97316": ORANGE, // bool
  "#94a3b8": token("muted-foreground"), // json
  "#cbd5e1": token("muted-foreground"), // any
  "#e2e8f0": LIGHT, // list
  "#ffffff": token("foreground"), // flow
  "#f472b6": token("chart-4"), // image
  "#fb7185": ROSE, // mask, heatmap
  "#34d399": token("chart-2"), // audio
  "#f87171": token("destructive"), // video
  "#60a5fa": token("chart-3"), // dataset
  "#38bdf8": INDIGO, // model
  "#2dd4bf": token("primary"), // checkpoint
  "#4ade80": token("chart-2"), // world
  "#22d3ee": TEAL, // table
  "#a3e635": LIME, // tensor
  "#64748b": token("muted-foreground"), // inputs
  "#8b5cf6": VIOLET, // text & LLM, default group
  "#a855f7": VIOLET, // story game
  "#f59e0b": token("chart-5"), // logic
  "#d97706": ORANGE, // transform
  "#0d9488": TEAL, // knowledge, memory
  "#14b8a6": TEAL, // automation, chat
  "#06b6d4": TEAL, // tables
  "#ec4899": token("chart-4"), // image
  "#ef4444": token("destructive"), // video
  "#10b981": token("chart-2"), // audio
  "#22c55e": token("chart-2"), // world game
  "#3b82f6": token("chart-3"), // data & training, recipe group
  "#0ea5e9": INDIGO, // output
};

/** The Studio token for a catalog colour; colours it doesn't know pass through. */
export function studioInk(color: string): string {
  return STUDIO_INK[color.toLowerCase()] ?? color;
}

/** A port/wire colour that stays readable on both themes (see `.flows-editor` in index.css). */
export function portInk(color: string): string {
  return `color-mix(in oklab, ${studioInk(color)} var(--flows-port-mix, 100%), black)`;
}
