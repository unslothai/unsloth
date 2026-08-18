// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// speculativeType and gpuMemoryMode describe the model that is actually running, and every
// writer sets them together with a loaded* shadow. The settings GET resolves after the
// first inference status on a cold boot, so hydrating the editable half while a shadow
// holds the other splits the pair. With nothing resident the opposite is true: the load
// path reads the store field (use-chat-model-runtime captures stateBeforeUnload), sends it
// and then persists it back, so a skipped hydration writes the local default over the
// server's preference. The predicate is tested for real here; the store itself cannot be
// imported (a .tsx barrel in its graph), so the wiring is pinned against the source.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { loadShadowOwnsMirroredSetting } from "../src/features/chat/utils/mirrored-chat-settings.ts";

const store = readFileSync(
  new URL("../src/features/chat/stores/chat-runtime-store.ts", import.meta.url),
  "utf8",
);

function slice(from: string, to: string): string {
  const start = store.indexOf(from);
  const end = store.indexOf(to, start + from.length);
  assert.ok(start !== -1, `not found: ${from}`);
  assert.ok(end !== -1, `not found: ${to}`);
  return store.slice(start, end);
}

const NO_MODEL = {
  loadedSpeculativeType: null,
  loadedGpuMemoryMode: null,
} as const;

// Dropping them instead would make scalarSettingMutationVersions[key] += 1 produce NaN and
// leave hydration permanently unable to match a version for either key.
test("both keys stay in the scalar setting list", () => {
  const keys = slice(
    "const SCALAR_SETTING_KEYS = [",
    "] as const satisfies readonly ScalarSettingKey[];",
  );
  assert.match(keys, /"speculativeType",/);
  assert.match(keys, /"gpuMemoryMode",/);
});

test("a resident model's shadow owns its half of the pair", () => {
  assert.equal(
    loadShadowOwnsMirroredSetting("speculativeType", {
      ...NO_MODEL,
      loadedSpeculativeType: "mtp",
    }),
    true,
  );
  assert.equal(
    loadShadowOwnsMirroredSetting("gpuMemoryMode", {
      ...NO_MODEL,
      loadedGpuMemoryMode: "manual",
    }),
    true,
  );
  // Each key answers for its own shadow only.
  assert.equal(
    loadShadowOwnsMirroredSetting("gpuMemoryMode", {
      ...NO_MODEL,
      loadedSpeculativeType: "mtp",
    }),
    false,
  );
});

// The regression: with no shadow, hydration must apply the server's value, or the next
// load captures the local default and the successful-load persist writes it back.
test("with nothing resident the stored preference still hydrates", () => {
  assert.equal(
    loadShadowOwnsMirroredSetting("speculativeType", NO_MODEL),
    false,
  );
  assert.equal(loadShadowOwnsMirroredSetting("gpuMemoryMode", NO_MODEL), false);
});

test("every other mirrored setting hydrates unconditionally", () => {
  for (const key of ["permissionMode", "ragMode", "toolsEnabled"]) {
    assert.equal(
      loadShadowOwnsMirroredSetting(key, {
        loadedSpeculativeType: "mtp",
        loadedGpuMemoryMode: "manual",
      }),
      false,
    );
  }
});

test("hydration defers to the shadow check, not the key name", () => {
  const hydrate = slice(
    "function getHydratedSettingsState(",
    "function setScalarSettingVersion<",
  );
  assert.match(hydrate, /loadShadowOwnsMirroredSetting\(key, state\)/);
  // An unconditional skip is the bug: it strands the server's preference.
  assert.doesNotMatch(
    hydrate,
    /if \(key === "speculativeType" \|\| key === "gpuMemoryMode"\)/,
  );
  // The check has to precede the write, or the field moves before anything can stop it.
  assert.ok(
    hydrate.indexOf("loadShadowOwnsMirroredSetting") <
      hydrate.indexOf("[key] = value;"),
    "the shadow check runs after hydration has already written the field",
  );
});

// The load path reads the standing preference from localStorage on a model switch, which
// cacheHydratedSettings refreshes from the same payload.
test("the mirrored cache still carries both preferences", () => {
  const mirrored = slice("const MIRRORED_SETTINGS = {", "\n} satisfies Partial<");
  assert.match(mirrored, /speculativeType: \{ storageKey: CHAT_SPECULATIVE_TYPE_KEY/);
  assert.match(mirrored, /gpuMemoryMode: \{ storageKey: CHAT_GPU_MEMORY_MODE_KEY/);
  assert.match(store, /loadString\(CHAT_SPECULATIVE_TYPE_KEY, "auto"\)/);
  assert.match(store, /loadString\(CHAT_GPU_MEMORY_MODE_KEY, "auto"\)/);
});

// saveBool reaches mirrorSettingToBackend, which bumps the version and queues the patch
// itself, so the sibling mirrored setters call it alone.
test("the artifact setters bump their setting version once", () => {
  const setters = slice(
    "setCollapseHtmlArtifacts: (collapseHtmlArtifacts) =>",
    "setMcpEnabledForChat: (mcpEnabledForChat) =>",
  );
  assert.doesNotMatch(setters, /setScalarSettingVersion/);
  assert.match(setters, /saveBool\(CHAT_COLLAPSE_HTML_ARTIFACTS_KEY/);
  assert.match(setters, /saveBool\(\s*CHAT_ALLOW_ARTIFACT_NETWORK_ACCESS_KEY/);
  assert.match(
    slice("function mirrorSettingToBackend(", "\n}"),
    /scalarSettingMutationVersions\[setting\.field\] \+= 1;/,
  );
});
