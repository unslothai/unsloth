// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

function source(relativePath: string): string {
  return readFileSync(
    new URL(`../src/features/chat/${relativePath}`, import.meta.url),
    "utf8",
  );
}

test("all max-output cap callers pass the selected connection override", () => {
  const settings = source("chat-settings-sheet.tsx");
  const runtime = source("stores/chat-runtime-store.ts");
  const adapter = source("api/chat-adapter.ts");

  assert.match(
    settings,
    /getExternalMaxOutputTokens\([\s\S]*?activeExternalProvider\?\.maxOutputTokens/,
  );
  assert.match(
    runtime,
    /getExternalMaxOutputTokens\([\s\S]*?provider\?\.maxOutputTokens/,
  );
  assert.match(
    adapter,
    /getExternalMaxOutputTokens\([\s\S]*?externalProvider\?\.maxOutputTokens/,
  );
});

test("the generic Custom editor exposes a bounded optional cap and warning", () => {
  const dialog = source("chat-providers-dialog.tsx");

  assert.match(dialog, /providerType === LEGACY_CUSTOM_PROVIDER_TYPE/);
  assert.match(dialog, /Max Tokens limit/);
  assert.match(dialog, /Leave blank to use the 32,768-token default\./);
  assert.match(
    dialog,
    /If the upstream provider does not support this value,\s+requests may fail\./,
  );
  assert.match(dialog, /min=\{CUSTOM_MAX_OUTPUT_TOKENS_MIN\}/);
  assert.match(dialog, /Number\.isSafeInteger\(value\)/);
  assert.match(dialog, /max=\{Number\.MAX_SAFE_INTEGER\}/);
});

test("preset application clamps live Max Tokens to the active external cap", () => {
  const settings = source("chat-settings-sheet.tsx");

  assert.match(
    settings,
    /function applyPresetParamsWithinCurrentLimits\([\s\S]*?if \(!isExternalModel\) return nextParams;[\s\S]*?Math\.min\(nextParams\.maxTokens, maxTokensMax\)/,
  );
  assert.match(
    settings,
    /onParamsChange\(applyPresetParamsWithinCurrentLimits\(p\.params\)\)/,
  );
  assert.match(
    settings,
    /applyPresetParamsWithinCurrentLimits\(fallbackPreset\.params\)/,
  );
});

test("lowering an active external cap immediately clamps live Max Tokens", () => {
  const settings = source("chat-settings-sheet.tsx");

  assert.match(
    settings,
    /useEffect\(\(\) => \{[\s\S]*?isExternalModel[\s\S]*?params\.maxTokens <= maxTokensMax[\s\S]*?maxTokens: maxTokensMax[\s\S]*?setActivePresetSource\(nextSource\)[\s\S]*?onParamsChange\(nextParams\)/,
  );
});
