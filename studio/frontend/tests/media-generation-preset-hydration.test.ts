// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const source = readFileSync(
  new URL(
    "../src/features/generation-presets/use-media-generation-presets.ts",
    import.meta.url,
  ),
  "utf8",
);
const imagesSource = readFileSync(
  new URL("../src/features/images/images-page.tsx", import.meta.url),
  "utf8",
);
const videoSource = readFileSync(
  new URL("../src/features/video/video-page.tsx", import.meta.url),
  "utf8",
);

test("saved load settings hydrate without overwriting an early user edit", () => {
  assert.match(
    source,
    /configKey\(currentLoadConfigRef\.current\) === initialLoadConfigKey\.current[\s\S]*applyLoadConfigRef\.current\(/,
  );
  assert.match(
    source,
    /settings\.activePresetSource === "modified"[\s\S]*settings\.currentLoadConfig \?\? undefined/,
  );
});

test("saved preset hydration preserves its canonical definition unless modified", () => {
  assert.match(source, /const currentDefaultParams = baselineParamsRef\.current/);
  assert.match(
    source,
    /normalized\.find\(\(preset\) => preset\.name === selected\)[\s\S]*params: currentDefaultParams[\s\S]*settings\.activePresetSource === "modified"[\s\S]*\? settings\.currentParams[\s\S]*: definition\.params/,
  );
});

test("video persistence owns canonical intent rather than the mapped controls", () => {
  assert.match(videoSource, /width: resolution\[0\][\s\S]*durationSeconds: durationIntentSeconds/);
  assert.match(videoSource, /setResolutionIntent\(\[params\.width, params\.height\]\)[\s\S]*setDurationIntentSeconds\(params\.durationSeconds\)[\s\S]*return params;/);
  assert.doesNotMatch(videoSource, /activePresetSourceRef\.current === "builtin-default"\)[\s\S]{0,160}setResolutionIntent/);
  assert.match(videoSource, /flowShift,[\s\S]*audioFlowShift,[\s\S]*setFlowShift\(params\.flowShift\)[\s\S]*setAudioFlowShift\(params\.audioFlowShift\)/);
  assert.doesNotMatch(videoSource, /setFlowShift\(defaultFlowShift\)/);
  assert.match(videoSource, /defaultFlowShift != null && flowShift != null[\s\S]*\{defaultFlowShift != null && \([\s\S]*value=\{flowShift \?\? defaultFlowShift\}/);
});

test("image generation presets stay out of workflow-specific forms", () => {
  assert.match(imagesSource, /workflow === "create" && \([\s\S]*<MediaGenerationPresetControl/);
});

test("the current load settings participate in persistence and modified state", () => {
  assert.match(source, /currentLoadConfig: currentLoadConfig \?\? null/);
  assert.match(
    source,
    /configKey\(currentLoadConfig\) === configKey\(baselineLoadConfigRef\.current\)/,
  );
});

test("a failed preset read never enables the autosave path", () => {
  const rejection = source.match(
    /\.catch\(\(\) => \{([\s\S]*?)\n\s*\}\);\n\s*return \(\) =>/,
  );
  assert.ok(rejection, "preset GET rejection handler must remain identifiable");
  assert.doesNotMatch(rejection[1]!, /setHydrated\(true\)/);
});

test("unload saves and named mutations start without waiting on a queue", () => {
  assert.match(source, /if \(keepalive\) \{\s*return operation\(\);/);
  assert.doesNotMatch(source, /presetMutationQueues/);
});

test("hydration reports whether settings came from persisted storage", () => {
  assert.match(source, /setHydrationSource\("fresh"\)/);
  assert.match(source, /setHydrationSource\("saved"\)/);
  assert.match(source, /const hydrated = hydrationSource !== "pending"/);
  assert.match(source, /const hasPersistedSettings = hydrationSource === "saved"/);
});

test("resident status does not replace saved pending load settings on refresh", () => {
  for (const pageSource of [imagesSource, videoSource]) {
    assert.match(pageSource, /const initialResolvedSeedHandled = useRef\(false\)/);
    assert.match(
      pageSource,
      /!\w+Presets\.hydrated[\s\S]*busy === "loading"[\s\S]*busy === "unloading"[\s\S]*!record/,
    );
    assert.match(
      pageSource,
      /!initialResolvedSeedHandled\.current[\s\S]*hasPersistedSettings && !canReapply/,
    );
  }
});

test("programmatic no-ops cannot leave a stale preset-source marker", () => {
  assert.doesNotMatch(source, /programmaticParamsKeyRef/);
  assert.match(
    source,
    /const nextSource =[\s\S]*key === paramsKey\(baselineParamsRef\.current\)[\s\S]*configKey\(currentLoadConfig\) === configKey\(baselineLoadConfigRef\.current\)/,
  );
});

test("optimistic model defaults update the selectable Default definition", () => {
  assert.match(source, /defaultParamsRef\.current = nextDefault/);
  assert.match(source, /setEffectiveDefaultParams\(nextDefault\)/);
  assert.match(
    source,
    /activePreset === DEFAULT_PRESET_NAME[\s\S]*params: effectiveDefaultParams/,
  );
  assert.doesNotMatch(source, /defaultParamsRef\.current = defaultParams;/);
});
