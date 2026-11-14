// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

const vars = new Map<string, string>();
(globalThis as { document?: unknown }).document = {
  documentElement: {
    style: {
      setProperty: (name: string, value: string) => vars.set(name, value),
      removeProperty: (name: string) => vars.delete(name),
    },
    setAttribute: () => undefined,
    removeAttribute: () => undefined,
    toggleAttribute: () => undefined,
    classList: { toggle: () => undefined },
  },
};

const { applyCustomizationToDocument, DEFAULT_CUSTOMIZATION } = await import(
  "../src/features/settings/stores/appearance-custom-store.ts"
);
const { COLOR_THEMES, FLAVOR_THEME_IDS } = await import(
  "../src/features/settings/lib/color-themes.ts"
);

function luminance(hex: string): number {
  const channel = (i: number) => {
    const c = Number.parseInt(hex.slice(i, i + 2), 16) / 255;
    return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  };
  return 0.2126 * channel(1) + 0.7152 * channel(3) + 0.0722 * channel(5);
}

function mix(hex: string, target: string, amount: number): string {
  const channel = (i: number) => {
    const from = Number.parseInt(hex.slice(i, i + 2), 16);
    const to = Number.parseInt(target.slice(i, i + 2), 16);
    return Math.round(from + (to - from) * amount)
      .toString(16)
      .padStart(2, "0");
  };
  return `#${channel(1)}${channel(3)}${channel(5)}`;
}

function ratio(a: string, b: string): number {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x) as [
    number,
    number,
  ];
  return (hi + 0.05) / (lo + 0.05);
}

const withAccent = (accent: string) => ({
  ...DEFAULT_CUSTOMIZATION,
  colors: {
    light: { ...DEFAULT_CUSTOMIZATION.colors.light, accent },
    dark: { ...DEFAULT_CUSTOMIZATION.colors.dark, accent },
  },
});

test("custom accents stay readable on each flavor theme's own surfaces", () => {
  for (const id of FLAVOR_THEME_IDS) {
    for (const mode of ["light", "dark"] as const) {
      const { background, surface } = COLOR_THEMES[id][mode];
      assert.ok(surface, `${id} ${mode} has no surface`);
      for (const accent of ["#aaaa33", "#22c55e", "#fde68a", "#101010"]) {
        applyCustomizationToDocument(withAccent(accent), mode, id);
        const corrected = vars.get("--primary") ?? "";
        const planes = [background, surface, mix(surface, corrected, 0.2)];
        for (const plane of planes) {
          assert.ok(
            ratio(corrected, plane) >= 2.5,
            `${accent} -> ${corrected} is ${ratio(corrected, plane).toFixed(2)}:1 on ${id} ${mode} ${plane}`,
          );
        }
      }
    }
  }
});
