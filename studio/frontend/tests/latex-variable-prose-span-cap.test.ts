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

test("short variable spans and math are unchanged", () => {
  assert.equal(preprocessLaTeX("Set $HOME/bin before $PATH and run it."), "Set &#36;HOME/bin before $PATH and run it.");
  assert.equal(preprocessLaTeX("Energy $E = mc^2$ holds."), "Energy $E = mc^2$ holds.");
});
