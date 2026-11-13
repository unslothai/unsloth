// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useState } from "react";

type ResolvedTheme = "light" | "dark";
type Rgba = [number, number, number, number];

/** Mirrors the --element-outline fallbacks on the glass rings in index.css. */
export const GLASS_ELEMENT_OUTLINE: Record<ResolvedTheme, number> = {
  light: 12,
  dark: 14,
};

/** The border as a slider value: how far it sits from the background towards the text colour. */
export function outlineStrength(
  border: Rgba,
  background: Rgba,
  foreground: Rgba,
): number {
  const alpha = border[3];
  let along = 0;
  let span = 0;
  for (let i = 0; i < 3; i++) {
    const seen = alpha * border[i] + (1 - alpha) * background[i];
    const axis = foreground[i] - background[i];
    along += (seen - background[i]) * axis;
    span += axis * axis;
  }
  if (span === 0) {
    return 0;
  }
  return Math.min(100, Math.max(0, Math.round((100 * along) / span)));
}

// Canvas resolves any CSS colour (oklch, color-mix, color()) to sRGB, which getComputedStyle does not.
function readColors(tokens: string[]): Rgba[] | null {
  const context = document
    .createElement("canvas")
    .getContext("2d", { willReadFrequently: true });
  if (!context) {
    return null;
  }
  const probe = document.createElement("span");
  probe.style.display = "none";
  document.body.append(probe);
  try {
    return tokens.map((token) => {
      probe.style.color = `var(${token})`;
      context.clearRect(0, 0, 1, 1);
      context.fillStyle = "#000";
      context.fillStyle = getComputedStyle(probe).color;
      context.fillRect(0, 0, 1, 1);
      const [r, g, b, a] = context.getImageData(0, 0, 1, 1).data;
      return [r, g, b, a / 255];
    });
  } finally {
    probe.remove();
  }
}

function readThemeOutline(): number | null {
  const root = document.documentElement;
  const resolved: ResolvedTheme = root.classList.contains("dark")
    ? "dark"
    : "light";
  const glassEdges =
    root.hasAttribute("data-glass") &&
    (root.hasAttribute("data-chat-wallpaper-workspace-ready") ||
      root.getAttribute("data-glass") === "macos");
  if (glassEdges) {
    return GLASS_ELEMENT_OUTLINE[resolved];
  }
  const colors = readColors(["--border", "--background", "--foreground"]);
  return colors ? outlineStrength(colors[0], colors[1], colors[2]) : null;
}

/** The current theme's own border strength, re-read whenever the theme, palette or glass changes. */
export function useThemeElementOutline(active: boolean): number | null {
  const [value, setValue] = useState<number | null>(null);
  useEffect(() => {
    if (!active) {
      return;
    }
    const update = () => setValue(readThemeOutline());
    update();
    const observer = new MutationObserver(update);
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: [
        "class",
        "style",
        "data-palette",
        "data-glass",
        "data-chat-wallpaper-workspace-ready",
        "data-contrast-adjust",
      ],
    });
    return () => observer.disconnect();
  }, [active]);
  return value;
}
