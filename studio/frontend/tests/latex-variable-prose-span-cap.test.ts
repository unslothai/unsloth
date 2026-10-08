// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { preprocessLaTeX } from "../src/lib/latex.ts";

test("a long dollar span does not freeze the variable-prose check", () => {
  const started = performance.now();
  preprocessLaTeX(`$${"A".repeat(2000)}$x`);
  assert.ok(performance.now() - started < 1000, "variable-prose regex backtracked on a long span");
});

test("many long dollar spans in one reply stay within the per-message budget", () => {
  const started = performance.now();
  preprocessLaTeX(`$${"A".repeat(256)}$x `.repeat(100) + `$${"A".repeat(128)}$x `.repeat(1000));
  assert.ok(performance.now() - started < 1000, "variable-prose regex work is not bounded per message");
});

test("short variable spans and math are unchanged", () => {
  assert.equal(preprocessLaTeX("Set $HOME/bin before $PATH and run it."), "Set &#36;HOME/bin before $PATH and run it.");
  assert.equal(preprocessLaTeX("Energy $E = mc^2$ holds."), "Energy $E = mc^2$ holds.");
});

test("dollar spans inside raw-text elements do not use up the budget", () => {
  const style = `<style>${`$${"A".repeat(127)}$x `.repeat(70)}</style>\n\n`;
  assert.ok(preprocessLaTeX(`${style}Set $HOME/bin before $PATH and run it.`).endsWith("Set &#36;HOME/bin before $PATH and run it."));
});
