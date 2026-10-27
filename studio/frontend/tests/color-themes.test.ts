// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  COLOR_THEMES,
  COLOR_THEME_IDS,
  FLAVOR_THEME_IDS,
  isColorThemeId,
} from "../src/features/settings/lib/color-themes.ts";

const read = (path: string) =>
  readFileSync(new URL(path, import.meta.url), "utf8");

const indexCss = read("../src/index.css");
const themeBoot = read("../public/theme-boot.js");
const backendSettings = read("../../backend/routes/settings.py");

test("every non-standard theme is applied by the boot script", () => {
  for (const id of COLOR_THEME_IDS) {
    if (id === "standard") continue;
    assert.match(themeBoot, new RegExp(`"${id}"`), id);
  }
});

test("the backend accepts every theme id", () => {
  const literal = backendSettings.match(/palette: Literal\[([\s\S]*?)\]/);
  assert.ok(literal);
  const accepted = [...literal[1].matchAll(/"([^"]+)"/g)].map((m) => m[1]);
  assert.deepEqual([...accepted].sort(), [...COLOR_THEME_IDS].sort());
});

test("every flavor theme has light and dark seed blocks", () => {
  for (const id of FLAVOR_THEME_IDS) {
    assert.ok(
      indexCss.includes(`:root[data-palette="${id}"]:not(.dark) {`),
      `${id} light`,
    );
    assert.ok(
      indexCss.includes(`:root[data-palette="${id}"].dark {`),
      `${id} dark`,
    );
    assert.ok(
      indexCss.includes(`[data-palette="${id}"],`) ||
        indexCss.includes(`[data-palette="${id}"])`),
      `${id} mapping`,
    );
  }
});

test("every theme id has registry colors", () => {
  assert.deepEqual(
    Object.keys(COLOR_THEMES).sort(),
    [...COLOR_THEME_IDS].sort(),
  );
});

test("isColorThemeId rejects unknown values", () => {
  assert.equal(isColorThemeId("matcha"), true);
  assert.equal(isColorThemeId("dracula"), false);
  assert.equal(isColorThemeId(null), false);
});

function luminance(hex: string): number {
  const channel = (i: number) => {
    const c = Number.parseInt(hex.slice(i, i + 2), 16) / 255;
    return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  };
  return 0.2126 * channel(1) + 0.7152 * channel(3) + 0.0722 * channel(5);
}

function ratio(a: string, b: string): number {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x) as [
    number,
    number,
  ];
  return (hi + 0.05) / (lo + 0.05);
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

test("flavor accents used as text meet the accent-text floor", () => {
  for (const id of FLAVOR_THEME_IDS) {
    for (const [mode, selector] of [
      ["light", `:root[data-palette="${id}"]:not(.dark) {`],
      ["dark", `:root[data-palette="${id}"].dark {`],
    ] as const) {
      const blocks: string[] = [];
      for (let start = indexCss.indexOf(selector); start >= 0; start = indexCss.indexOf(selector, start + 1)) {
        blocks.push(indexCss.slice(start, indexCss.indexOf("}", start)));
      }
      assert.ok(blocks.length > 0, `${id} ${mode} block`);
      const block = blocks.join("\n");
      const token = (name: string) =>
        block.match(new RegExp(`\\s${name}: (#[0-9a-f]{6});`))?.[1];
      const page = token("--th-bg");
      const surface = token("--th-surface");
      const accent = token("--th-accent");
      assert.ok(page && surface && accent, `${id} ${mode} seeds`);
      for (const name of ["--primary", "--control-accent"]) {
        const color = token(name) ?? accent;
        for (const plane of [page, surface, mix(surface, color, 0.2)]) {
          assert.ok(
            ratio(color, plane) >= 2.5,
            `${id} ${mode} ${name} ${color} is ${ratio(color, plane).toFixed(2)}:1 on ${plane}`,
          );
        }
      }
    }
  }
});

test("More themes leads with Butterfly Pea and Earl Grey, and Cherry is just Cherry", () => {
  assert.deepEqual(FLAVOR_THEME_IDS.slice(0, 5), ["matcha", "espresso", "butterfly-pea", "cherry", "earl-grey"]);
  assert.equal(FLAVOR_THEME_IDS.indexOf("blueberry"), 11);
  assert.equal(FLAVOR_THEME_IDS.indexOf("honey"), 13);
  assert.equal(COLOR_THEMES.cherry.name, "Cherry");
});
