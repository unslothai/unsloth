// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// speculativeType and gpuMemoryMode hydrate only when no loaded* shadow holds the running value.
// The store imports a .tsx barrel, so its wiring is pinned against source.

import assert from "node:assert/strict";
import test from "node:test";

import { loadShadowOwnsMirroredSetting } from "../src/features/chat/utils/mirrored-chat-settings.ts";

import { readSrc } from "./helpers/kit.ts";

const store = readSrc("features/chat/stores/chat-runtime-store.ts");

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

// Dropping a key would make the version bump produce NaN and break hydration.
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
  assert.equal(
    loadShadowOwnsMirroredSetting("gpuMemoryMode", {
      ...NO_MODEL,
      loadedSpeculativeType: "mtp",
    }),
    false,
  );
});

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
  assert.doesNotMatch(
    hydrate,
    /if \(key === "speculativeType" \|\| key === "gpuMemoryMode"\)/,
  );
  assert.ok(
    hydrate.indexOf("loadShadowOwnsMirroredSetting") <
      hydrate.indexOf("[key] = value;"),
    "the shadow check runs after hydration has already written the field",
  );
});

test("the mirrored cache still carries both preferences", () => {
  const mirrored = slice("const MIRRORED_SETTINGS = {", "\n} satisfies Partial<");
  assert.match(mirrored, /speculativeType: \{ storageKey: CHAT_SPECULATIVE_TYPE_KEY/);
  assert.match(mirrored, /gpuMemoryMode: \{ storageKey: CHAT_GPU_MEMORY_MODE_KEY/);
  assert.match(store, /loadString\(CHAT_SPECULATIVE_TYPE_KEY, "auto"\)/);
  assert.match(store, /loadString\(CHAT_GPU_MEMORY_MODE_KEY, "auto"\)/);
});

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
