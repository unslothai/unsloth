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
    // Inside the `if (provider)` guard, so the optional chain is gone: an
    // unresolved provider must not clamp at all, rather than clamp to the fallback.
    /if \(provider\) \{[\s\S]*?getExternalMaxOutputTokens\([\s\S]*?provider\.maxOutputTokens/,
  );
  assert.match(
    adapter,
    /getExternalMaxOutputTokens\([\s\S]*?externalProvider\?\.maxOutputTokens/,
  );
});

test("the generic Custom editor exposes a bounded optional cap and warning", () => {
  const dialog = source("chat-providers-dialog.tsx");

  // Gated on the extracted predicate, not on the UI provider type: line 136's
  // `providerType === LEGACY_CUSTOM_PROVIDER_TYPE` is a display-name lookup and
  // matching it here would pass even if the row lost its gate entirely.
  assert.match(
    dialog,
    /const supportsMaxOutputTokens = supportsCustomMaxOutputTokens\(/,
  );
  assert.match(dialog, /\{supportsMaxOutputTokens \? \(/);
  assert.match(dialog, /Max Tokens limit/);
  assert.match(dialog, /Leave blank to use the 32,768-token default\./);
  assert.match(
    dialog,
    /If the upstream provider does not support this value,\s+requests may fail\./,
  );
  // A TEXT input, not `type="number"`. The HTML value sanitization algorithm
  // replaces anything a `number` input does not read as a valid floating-point
  // number with the empty string, and empty means "no override" here, so a
  // grouped entry like "131,072" would silently CLEAR the user's override on
  // save. The bounds live in `parseMaxOutputTokens` instead of in DOM attributes.
  assert.match(
    dialog,
    /id="provider-max-output-tokens"\s+type="text"\s+inputMode="numeric"/,
  );
  assert.doesNotMatch(
    dialog,
    /id="provider-max-output-tokens"\s+type="number"/,
  );
  assert.match(dialog, /value < CUSTOM_MAX_OUTPUT_TOKENS_MIN/);
  assert.match(dialog, /Number\.isSafeInteger\(value\)/);
  assert.match(dialog, /\/\^\\d\+\$\/\.test\(trimmed\)/);
});

test("preset application clamps live Max Tokens to the active external cap", () => {
  const settings = source("chat-settings-sheet.tsx");

  assert.match(
    settings,
    // The provider guard is part of the shape: without a resolved provider the cap
    // is the 32,768 fallback, and applying a preset there would lower the value for
    // good. `custom-provider-max-output-tokens-guards.test.ts` covers the same rule
    // at the other two clamp sites.
    /function applyPresetParamsWithinCurrentLimits\([\s\S]*?if \(!isExternalModel \|\| activeExternalProvider == null\) return nextParams;[\s\S]*?Math\.min\(nextParams\.maxTokens, maxTokensMax\)/,
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

  // The decision itself lives in `resolveExternalMaxTokensClamp` (unit-tested in
  // external-max-tokens-clamp.test.ts). What this asserts is that the effect still
  // asks it, still passes the availability inputs rather than assuming them, and
  // still writes the result back through the preset-source bookkeeping.
  assert.match(
    settings,
    /useEffect\(\(\) => \{\s*const clampedMaxTokens = resolveExternalMaxTokensClamp\(\{[\s\S]*?settingsHydrated,[\s\S]*?hasActiveExternalProvider: activeExternalProvider != null,[\s\S]*?isExternalModel,[\s\S]*?maxTokens: params\.maxTokens,[\s\S]*?maxTokensMax,[\s\S]*?\}\);[\s\S]*?if \(clampedMaxTokens == null\) \{[\s\S]*?maxTokens: clampedMaxTokens[\s\S]*?setActivePresetSource\(nextSource\)[\s\S]*?onParamsChange\(nextParams\)/,
  );
});
