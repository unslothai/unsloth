// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { groupStateKeys, keyedGroupStates } from "../vite-plugin-group-state.ts";

test("group hover, focus-within and active descendants are keyed on the group class", () => {
  assert.equal(
    keyedGroupStates(".group-hover\\:opacity-100:is(:where(.group):hover *){opacity:1}"),
    ".group-hover\\:opacity-100:is(.group:hover *){opacity:1}",
  );
  assert.equal(
    keyedGroupStates(".a:is(:where(.group\\/assistant-message):focus-within *),.b:is(:where(.group\\/row):active *)"),
    ".a:is(.group\\/assistant-message:focus-within *),.b:is(.group\\/row:active *)",
  );
});

test("other :where() forms are left alone", () => {
  for (const css of [
    ".peer-hover\\:x:is(:where(.peer):hover~*){}",
    ".x:is(:where(.group):focus-visible *){}",
    ".x:is(:where(.group):has(input:disabled) *){}",
    ":where(.a > *):where(:not(:last-child)){}",
    ".x:where(.dark, .dark *){}",
  ]) {
    assert.equal(keyedGroupStates(css), css);
  }
});

test("idempotent", () => {
  const once = keyedGroupStates(".x:is(:where(.group\\/n):hover *)");
  assert.equal(keyedGroupStates(once), once);
});

test("the plugin runs on CSS after Tailwind in the shipped config", () => {
  const hook = groupStateKeys().transform as { handler: (code: string) => { code: string } | null };
  assert.equal(hook.handler(".x{}"), null);
  assert.equal(hook.handler(".x:is(:where(.group):hover *){}")?.code, ".x:is(.group:hover *){}");
  const vite = readFileSync(new URL("../vite.config.ts", import.meta.url), "utf8");
  assert.match(vite, /tailwindcss\(\), childSelectorOrder\(\), hasAnchorOrder\(\), viewportMediaGate\(\), groupStateKeys\(\),/);
});

test("a disabled number field keeps its stepper hidden against the group hover / focus-within reveal", () => {
  const input = readFileSync(new URL("../src/components/ui/input.tsx", import.meta.url), "utf8");
  assert.match(input, /group-has-\[input:disabled\]\/number:opacity-0!/);
});
