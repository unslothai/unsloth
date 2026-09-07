// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// "Only unload models loaded by the API" vetoes the media TTL outright: every resident image
// or video model is one the user loaded from Unsloth, so there is nothing the setting would
// let go of. The switch that lifts the veto used to render only while the CHAT idle unload
// was active, so a user who had turned that off after enabling the option saw the media row
// go straight to "paused" with no control anywhere to explain it or undo it -- the feature
// was unusable without re-enabling chat unloading first.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

const SECTION = readFileSync(
  fileURLToPath(
    new URL(
      "../src/features/settings/components/model-auto-switch-section.tsx",
      import.meta.url,
    ),
  ),
  "utf8",
);

// The JSX guard the given row is rendered under: the nearest conditional above it at the
// section's own indentation, so a `{settings.foo}` prop inside a neighbouring row is not
// mistaken for one.
const GUARD = "\n      {settings";

function guardFor(labelKey: string): string {
  const upTo = SECTION.slice(0, SECTION.indexOf(`modelAutoSwitch.${labelKey}"`));
  return upTo.slice(upTo.lastIndexOf(GUARD));
}

test("the API-only switch is reachable whenever a media TTL is saved", () => {
  assert.match(guardFor("apiOnly"), /mediaAutoUnloadIdleSeconds > 0/);
});

test("it is still reachable from the chat TTL alone", () => {
  // The media TTL is off by default, so this stays the ordinary way in.
  assert.match(guardFor("apiOnly"), /idleUnloadActive/);
});

test("the KV-save option stays with the chat TTL it belongs to", () => {
  // It persists llama.cpp slot KV; there is no media equivalent, so widening the API-only
  // row must not drag it along.
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
