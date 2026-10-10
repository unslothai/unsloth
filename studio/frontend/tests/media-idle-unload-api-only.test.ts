// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The API-only switch must render without chat idle unload, or the media TTL row is stuck paused.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const SECTION = readSrc("features/settings/components/model-auto-switch-section.tsx");

const GUARD = "\n      {settings";

function guardFor(labelKey: string): string {
  const upTo = SECTION.slice(0, SECTION.indexOf(`modelAutoSwitch.${labelKey}"`));
  return upTo.slice(upTo.lastIndexOf(GUARD));
}

test("the API-only switch is reachable whenever a media TTL is saved", () => {
  assert.match(guardFor("apiOnly"), /mediaAutoUnloadIdleSeconds > 0/);
});

test("it is still reachable from the chat TTL alone", () => {
  assert.match(guardFor("apiOnly"), /idleUnloadActive/);
});

test("the KV-save option stays with the chat TTL it belongs to", () => {
  // keepKv is llama.cpp slot KV only; no media equivalent.
  const guard = guardFor("keepKv");
  assert.match(guard, /idleUnloadActive/);
  assert.doesNotMatch(guard, /mediaAutoUnloadIdleSeconds/);
});

test("the media row still says when a veto is holding its TTL", () => {
  assert.match(
    SECTION,
    /settings\.mediaAutoUnloadIdleSeconds > 0 &&\s*!settings\.mediaIdleUnloadActive/,
  );
});
