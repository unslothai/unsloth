// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The live Interpretability tab gated on the editable form store, so a run started
// through the API with capture on (or after the form changed) read as "disabled" and
// never polled. It reads the active run's saved config first, as the config popover does.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

test("the saved run config carries the capture flag", () => {
  const src = readSrc("features/studio/sections/run-config-override.ts");
  assert.match(src, /enableActivationCapture\?: boolean;/);
  assert.match(
    src,
    /enableActivationCapture: config\.enable_activation_capture === true,/,
  );
});

test("the live tab prefers the run's capture flag over the form store", () => {
  const src = readSrc("features/studio/live-training-view.tsx");
  assert.match(
    src,
    /\(runConfigOverride\?\.enableActivationCapture \?\?\s+\(config\.enableActivationCapture && !config\.isAudioModel\)\) \?/,
  );
  assert.doesNotMatch(src, /\{config\.enableActivationCapture \? \(/);
});
