// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { promptUsesHighPrecisionTimeVariables } = await import(
  "../src/features/chat/api/prompt-time-variables.ts"
);

test("second-precision time variables warn (#9177)", () => {
  for (const prompt of [
    "The current time is {{$now}}",
    "Clock: {{ $time }}",
    "Context as of {{$now}}, also {{version}}.",
  ]) {
    assert.equal(promptUsesHighPrecisionTimeVariables(prompt), true, prompt);
  }
});

test("variables that do not change per request stay quiet", () => {
  for (const prompt of [
    "{{ $NOW }}",
    "Today is {{$date}}",
    "Env: {{ env }}",
    "see {{$timestamp}}",
    "Plain prompt",
    "",
  ]) {
    assert.equal(promptUsesHighPrecisionTimeVariables(prompt), false, prompt);
  }
});
