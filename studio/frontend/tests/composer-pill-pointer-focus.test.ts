// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { watchInputModality } from "../src/lib/input-modality.ts";

const CSS = await readFile(new URL("../src/index.css", import.meta.url), "utf8");
const MAIN = await readFile(new URL("../src/main.tsx", import.meta.url), "utf8");

function fakeWindow() {
  const listeners = new Map<string, (event: { key?: string }) => void>();
  const root = { dataset: {} as Record<string, string> };
  const win = {
    document: { documentElement: root },
    addEventListener: (type: string, fn: (event: { key?: string }) => void) => listeners.set(type, fn),
  } as unknown as Window;
  return { win, root, fire: (type: string, event = {}) => listeners.get(type)?.(event) };
}

test("tracks whether the last input was a pointer or the keyboard", () => {
  const { win, root, fire } = fakeWindow();
  watchInputModality(win);
  fire("pointerdown");
  assert.equal(root.dataset.inputModality, "pointer");
  fire("keydown", { key: "Shift" });
  assert.equal(root.dataset.inputModality, "pointer");
  fire("keydown", { key: "Tab" });
  assert.equal(root.dataset.inputModality, "keyboard");
  assert.match(MAIN, /watchInputModality\(window\);/);
});

test("composer pills drop the focus ring after a click but keep it for the keyboard", () => {
  assert.match(CSS, /\.composer-pill-btn:focus-visible,[\s\S]*?box-shadow: 0 0 0 1px var\(--ring-soft\);/);
  assert.match(
    CSS,
    /html\[data-input-modality="pointer"\]\s*:is\(\.composer-pill-btn, \.unsloth-thinking-pill, \.unsloth-composer-plus\):focus-visible \{\s*box-shadow: none;/,
  );
});

test("pill labels are not selectable", () => {
  assert.match(CSS, /\.composer-pill-btn \{\s*@apply [^;]*\bselect-none\b/);
});
