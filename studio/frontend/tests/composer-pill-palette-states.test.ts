// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { COLOR_THEME_IDS } from "../src/features/settings/lib/color-themes.ts";
import { readSrc } from "./helpers/kit.ts";

// Dark composer pills rest at full ink and turn var(--primary) when on, so a dark
// palette whose primary sits next to the ink needs its off pills dimmed (#12025).

const CSS = readSrc("index.css");
const TOKEN = /(--[\w-]+):\s*([^;]+);/g;
const HEX = /^#[0-9a-f]{6}$/i;
const PALETTE_ATTR = /data-palette="([\w-]+)"/g;
const DIM_RULE =
  /\n\t:root:is\(([^)]*)\)\.dark\s+:is\(\.composer-pill-btn, \.unsloth-thinking-pill\)\[data-active="false"\] \{\s*color: color-mix\(in oklab, var\(--foreground\) 60%, transparent\);/;
// Classic and Minimal dark sit near 0.04 (OKLab), every other dark palette above 0.13.
const COLLIDES = 0.1;

function darkTokens(selector: string): Map<string, string> {
  const tokens = new Map<string, string>();
  const escaped = selector.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  for (const block of CSS.matchAll(
    new RegExp(`\\n${escaped} \\{([^}]*)\\}`, "g"),
  )) {
    for (const [, name, value] of block[1].matchAll(TOKEN)) {
      tokens.set(name, value.trim());
    }
  }
  return tokens;
}

function oklab(hex: string): number[] {
  const [r, g, b] = [1, 3, 5].map((i) => {
    const c = Number.parseInt(hex.slice(i, i + 2), 16) / 255;
    return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  });
  const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
  const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
  const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
  return [
    0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s,
    1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s,
    0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s,
  ];
}

const base = darkTokens(".dark");

/** OKLab distance between a dark palette's on ink (--primary) and its resting ink. */
function onOffDistance(id: string): number {
  const own =
    id === "standard" ? base : darkTokens(`:root[data-palette="${id}"].dark`);
  const primary =
    own.get("--primary") ??
    own.get("--th-accent") ??
    base.get("--primary") ??
    "";
  const ink =
    own.get("--foreground-base") ??
    own.get("--th-fg") ??
    base.get("--foreground-base") ??
    "";
  assert.match(primary, HEX, `${id} --primary`);
  assert.match(ink, HEX, `${id} ink`);
  const [a, b] = [oklab(primary), oklab(ink)];
  return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

const dimmed = new Set(
  [...(DIM_RULE.exec(CSS)?.[1] ?? "").matchAll(PALETTE_ATTR)].map((m) => m[1]),
);

test("every dark palette with an on pill close to the resting ink dims its off pills", () => {
  const colliding = COLOR_THEME_IDS.filter(
    (id) => onOffDistance(id) < COLLIDES,
  );
  assert.ok(
    colliding.includes("classic"),
    "the measure no longer sees the Classic collision",
  );
  for (const id of colliding) {
    assert.ok(dimmed.has(id), `${id} dark: off pills not dimmed`);
  }
});

test("palettes whose accent already stands apart keep full-ink off pills", () => {
  for (const id of dimmed) {
    assert.ok(
      onOffDistance(id) < COLLIDES,
      `${id} is dimmed without needing it`,
    );
  }
});
