// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { CPU_FALLBACK_MESSAGE, loadFallbackNotice } = await import(
  "../src/features/chat/utils/mmproj-fallback.ts"
);

// Hoisted: biome's useTopLevelRegex flags a literal recompiled per call.
const ON_CPU_SUFFIX = / on CPU$/;
const GPU_DISABLED = /GPU acceleration is disabled for this model session/;
const DELEGATES = /loadFallbackNotice\(/;
const PICKS_WARNING = /notice\.degraded \? toast\.warning : toast\.success/;
const PASSES_DESCRIPTION = /description: notice\.description/;
const FORWARDS_REASON = /cpu_fallback_reason/;
const CALL_SITES = /\n\s*showAutoLoadSuccess\([\s\S]*?\);/g;

// Asserted against the source like the other chat-adapter tests: importing the
// module would drag in the stores and the toast layer for one closure.
const source = readFileSync(
  fileURLToPath(
    new URL("../src/features/chat/api/chat-adapter.ts", import.meta.url),
  ),
  "utf8",
);

// The wording and the warn-vs-success choice used to be written inline in
// showAutoLoadSuccess, and this file matched them there by substring. Both now
// live in loadFallbackNotice, which is the one definition the explicit-load path
// shares, so the behaviour is asserted by calling it and the call site is
// asserted only to delegate. Matching the inline form again would go red on a
// refactor that changed nothing a user can see, and -- worse -- stay green if
// only the explicit-load path kept the behaviour.
test("an auto-load that fell back to CPU warns instead of claiming plain success", () => {
  const notice = loadFallbackNotice(
    "Loaded Qwen3 (Q4_K_M)",
    "vulkan_startup_crash",
    null,
  );
  assert.equal(notice.degraded, true, "a CPU fallback is not a plain success");
  assert.match(notice.title, ON_CPU_SUFFIX);
  assert.match(notice.description ?? "", GPU_DISABLED);
  assert.equal(notice.description, CPU_FALLBACK_MESSAGE);
});

test("a load with no fallback stays a plain success", () => {
  const notice = loadFallbackNotice("Loaded Qwen3 (Q4_K_M)", null, null);
  assert.equal(notice.degraded, false);
  assert.equal(notice.title, "Loaded Qwen3 (Q4_K_M)");
  assert.equal(notice.description, undefined);
});

test("the auto-load toast is driven by that verdict, not by its own copy of it", () => {
  const start = source.indexOf("const showAutoLoadSuccess = (");
  assert.ok(start >= 0, "showAutoLoadSuccess is no longer defined");
  const helper = source.slice(start, source.indexOf("\n  };", start));
  assert.match(helper, DELEGATES);
  assert.match(helper, PICKS_WARNING);
  assert.match(helper, PASSES_DESCRIPTION);
});

test("every auto-load path forwards the load response's fallback reason", () => {
  const calls = source.match(CALL_SITES) ?? [];
  assert.ok(calls.length > 0, "no showAutoLoadSuccess call sites found");
  for (const call of calls) {
    assert.match(call, FORWARDS_REASON);
  }
});
