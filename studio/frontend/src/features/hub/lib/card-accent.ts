// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { CSSProperties } from "react";

export type AdaptiveCardAccentStyle = CSSProperties & {
  "--hub-card-art-accent"?: string;
  "--hub-card-inner-mix"?: string;
  "--hub-card-inner-opacity"?: string;
};

interface Rgb {
  r: number;
  g: number;
  b: number;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

function formatPercent(value: number): string {
  return `${Math.round(value * 10) / 10}%`;
}

function formatOpacity(value: number): string {
  return (Math.round(value * 100) / 100).toString();
}

function parseRgb(color: string): Rgb | null {
  const value = color.trim().toLowerCase();
  if (!(value.startsWith("rgb(") || value.startsWith("rgba("))) {
    return null;
  }

  const start = value.indexOf("(");
  const end = value.indexOf(")", start + 1);
  if (start === -1 || end === -1) {
    return null;
  }

  const channels = value
    .slice(start + 1, end)
    .split(",")
    .slice(0, 3)
    .map((channel) => Number.parseFloat(channel.trim()));
  if (
    channels.length !== 3 ||
    channels.some((channel) => !Number.isFinite(channel))
  ) {
    return null;
  }

  const [r, g, b] = channels.map((channel) => clamp(channel, 0, 255));
  return { r, g, b };
}

function relativeLuminance({ r, g, b }: Rgb): number {
  const toLinear = (channel: number) => {
    const value = channel / 255;
    return value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4;
  };
  return 0.2126 * toLinear(r) + 0.7152 * toLinear(g) + 0.0722 * toLinear(b);
}

function saturation({ r, g, b }: Rgb): number {
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  return max === 0 ? 0 : (max - min) / max;
}

export function buildAdaptiveCardAccentStyle(
  color: string | null,
): AdaptiveCardAccentStyle | undefined {
  if (!color) {
    return undefined;
  }

  const rgb = parseRgb(color);
  if (!rgb) {
    return undefined;
  }

  const sat = saturation(rgb);
  if (sat < 0.22) {
    return undefined;
  }

  const lum = relativeLuminance(rgb);
  const darkPenalty = lum < 0.24 ? 3.4 : lum < 0.36 ? 2.1 : 0;
  const brightPenalty = lum > 0.78 ? 1.4 : 0;

  const innerMix = clamp(
    16 + sat * 9 - darkPenalty * 1.8 - brightPenalty,
    12,
    25,
  );
  const innerOpacity = clamp(
    0.18 + sat * 0.16 - darkPenalty * 0.035 - brightPenalty * 0.015,
    0.16,
    0.34,
  );

  return {
    "--hub-card-art-accent": color,
    "--hub-card-inner-mix": formatPercent(innerMix),
    "--hub-card-inner-opacity": formatOpacity(innerOpacity),
  };
}
