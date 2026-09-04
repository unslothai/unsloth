// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { promptUsesHighPrecisionTimeVariables } = await import(
  "../src/features/chat/api/prompt-time-variables.ts"
);

// #9177: {{$now}}/{{$time}} are re-filled with second precision on every
// request, so a system prompt carrying them changes every turn and prefix
// caching never matches — full prefill per message (55s vs 0.95s first token
// in the report). The UI warns on exactly this shape.
test("detects the high-precision time variables", () => {
  assert.equal(
    promptUsesHighPrecisionTimeVariables("The current time is {{$now}}"),
    true,
  );
  assert.equal(
    promptUsesHighPrecisionTimeVariables("Clock: {{ $time }}"),
    true,
  );
  // Case-sensitive: the substitution resolves $now/$time exactly, so an
  // uppercase variant is left unsubstituted and does not churn the prefix.
  assert.equal(
    promptUsesHighPrecisionTimeVariables("{{ $NOW }}"),
    false,
  );
});

test("day precision and custom variables do not warn", () => {
  assert.equal(
    promptUsesHighPrecisionTimeVariables("Today is {{$date}}"),
    false,
  );
  assert.equal(
    promptUsesHighPrecisionTimeVariables("Env: {{ env }}"),
    false,
  );
  assert.equal(
    promptUsesHighPrecisionTimeVariables("Plain prompt, no variables"),
    false,
  );
  assert.equal(promptUsesHighPrecisionTimeVariables(""), false);
});

test("now embedded mid-sentence and with timezone suffix still warns", () => {
  assert.equal(
    promptUsesHighPrecisionTimeVariables(
      "Context as of {{$now}} — respond accordingly. Also {{version}}.",
    ),
    true,
  );
  assert.equal(
    promptUsesHighPrecisionTimeVariables("see {{$timestamp}}"),
    false,
    "only the built-in $now/$time shapes warn; unknown keys are left as-is",
  );
});
