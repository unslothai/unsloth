// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// speculativeType and gpuMemoryMode describe the model that is actually running, and every
// writer sets them together with a loaded* shadow. The settings GET resolves after the
// first inference status on a cold boot, so hydrating the editable half on its own leaves
// the pair split: the repair guard in apply-inference-status-to-store only fires while the
// shadow is null, gpuMemoryEditsPending is the inequality itself, and both keys sit in the
// model-config editor's React key. The store cannot be imported here (a .tsx barrel in its
// graph), so these pin the source the way the other chat-runtime-store tests do.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

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

test("hydration skips the two keys a model load owns", () => {
  const hydrate = slice(
    "function getHydratedSettingsState(",
    "function setScalarSettingVersion<",
  );
  assert.match(
    hydrate,
    /if \(key === "speculativeType" \|\| key === "gpuMemoryMode"\) \{\s*continue;/,
  );
  // The skip has to precede the write, or the field moves before anything can stop it.
  assert.ok(
    hydrate.indexOf('key === "speculativeType"') <
      hydrate.indexOf("[key] = value;"),
    "the skip runs after hydration has already written the field",
  );
});

// The load path reads the standing preference from localStorage, which cacheHydratedSettings
// refreshes from the same payload, so skipping the store write loses no preference.
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
