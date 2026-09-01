// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const voiceTab = readFileSync(
  new URL("../src/features/settings/tabs/voice-tab.tsx", import.meta.url),
  "utf8",
);
const adapter = readFileSync(
  new URL(
    "../src/features/chat/adapters/studio-model-dictation-adapter.ts",
    import.meta.url,
  ),
  "utf8",
);

test("changing the dictation device releases only this tab's model", () => {
  // Unscoped, this races another surface: the unload lands after that surface
  // swapped the resident model and tears down one this tab never owned.
  assert.match(
    voiceTab,
    /void unloadSttModel\(sttEngineFor\(sttModel\), sttModel, \{\s*wait: false,\s*\}\)/,
  );
});

test("switching device never kills a transcription being decoded", () => {
  // The default drains the sidecar for 30s and then releases it anyway, which
  // throws the recording away. This release is only an early one: the next load
  // applies the setting regardless, so there is nothing to wait for.
  assert.match(voiceTab, /wait: false/);
  assert.match(adapter, /if \(options\?\.wait === false\) params\.set\("wait", "false"\)/);
});

test("the unload API still takes the engine and model that scoping needs", () => {
  assert.match(
    adapter,
    /export function unloadSttModel\(\s*engine\?: SttEngine,\s*model\?: string,\s*options\?: \{ wait\?: boolean \},\s*\)/,
  );
  assert.match(adapter, /if \(model\) params\.set\("model", model\)/);
});

test("the device preference travels with every load and transcribe", () => {
  // A load that omits it is read as "no opinion" server-side, so the setting
  // would silently never apply.
  assert.match(adapter, /device: resolvedDevice/);
  assert.match(adapter, /params\.set\("device", settings\.sttDevice\)/);
});
