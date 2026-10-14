// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const page = readFileSync(new URL("../src/features/images/images-page.tsx", import.meta.url), "utf8");

test("a pick refused while busy does not move the Images page to another machine", () => {
  const start = page.indexOf("const handleModelSelect = useCallback(");
  assert.ok(start >= 0);
  const body = page.slice(start);
  const guard = body.indexOf("if (busy !== null && !downloadOnlyPick) return;");
  const switchMachine = body.indexOf("setImagesMachine(machine)");
  assert.ok(guard >= 0 && switchMachine >= 0);
  assert.ok(guard < switchMachine, "the busy guard must run before the machine switch");
});

test("a download-only pick of a linked model is refused before it reaches this machine's downloads", () => {
  const body = page.slice(page.indexOf("const handleModelSelect = useCallback("));
  const refuse = body.indexOf("if (downloadOnlyPick && machine)");
  const switchMachine = body.indexOf("setImagesMachine(machine)");
  assert.ok(refuse >= 0 && refuse < switchMachine);
});

test("a deleted Images machine falls back to this machine once the linked list refreshes", () => {
  const effect = page.slice(page.indexOf("A machine removed since it was picked"));
  const resetAt = effect.indexOf("setImagesMachine(null)");
  assert.ok(resetAt > 0 && resetAt < effect.indexOf("}, [imagesMachine, refreshStatus]);"));
  assert.ok(effect.indexOf("instances.some((i) => i.id === imagesMachine)") < resetAt);
});
