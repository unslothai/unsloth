// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The sibling of vision-toggle-load-payload-parity.test.ts, and the half it does
// not cover. That test counts the snake_case keys going OUT in a /load payload.
// This one counts the camelCase keys coming BACK into the runtime store from the
// load echo.
//
// Payload parity was complete while store parity was not: every load path
// repaired `tensorParallel` from the response and left `disableVision` holding
// whatever the PREVIOUS model had. A background auto-load, a compare pane, or a
// failed switch that rolled back then showed Vision off in Advanced Settings over
// a server running with its projector loaded -- and the next Apply sent that
// phantom off and silently dropped the projector. That is the same failure
// apply-inference-status-to-store.ts exists to prevent on reload, arriving by a
// different door.
//
// Source-level for the same reason as the payload test: these modules reach the
// chat barrel, which re-exports JSX that --experimental-strip-types cannot compile.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const read = (path: string) =>
  readFileSync(new URL(`../${path}`, import.meta.url), "utf8");

const count = (haystack: string, needle: RegExp) =>
  (haystack.match(needle) ?? []).length;

// Hoisted for biome's useTopLevelRegex. The lookbehind keeps `loadedTensorParallel:`
// and `loadedVisionDisabledByUser:` -- the read-only baselines -- out of the count;
// only the EDITABLE knobs, the ones Advanced Settings renders, are compared.
const TENSOR_PARALLEL_STORE_WRITE = /(?<![A-Za-z])tensorParallel:/g;
const DISABLE_VISION_STORE_WRITE = /(?<![A-Za-z])disableVision:/g;

// Every module that writes the editable load knobs back into the runtime store
// after a load, rollback or auto-load. tensorParallel is the reference for the
// same reason as in the payload test: identical lifecycle, always repaired.
const STORE_WRITERS = [
  "src/features/chat/api/chat-adapter.ts",
  "src/features/chat/shared-composer.tsx",
  "src/features/chat/hooks/use-chat-model-runtime.ts",
];

for (const path of STORE_WRITERS) {
  test(`${path} repairs the toggle everywhere it repairs tensorParallel`, () => {
    const src = read(path);
    const tp = count(src, TENSOR_PARALLEL_STORE_WRITE);
    const dv = count(src, DISABLE_VISION_STORE_WRITE);
    assert.ok(tp > 0, "reference knob vanished; this test needs re-anchoring");
    assert.equal(
      dv,
      tp,
      `${path} writes tensorParallel to the store ${tp} time(s) but disableVision ${dv}: a load path leaves the Vision switch showing the previous model's state`,
    );
  });
}

test("the api-monitor override summary names the toggle", () => {
  // The saved-override summary enumerates what a remote load will apply. Omitting
  // Vision means a model saved with the projector off reads as if it were on.
  const src = read(
    "src/features/api-monitor/components/saved-model-settings.tsx",
  );
  assert.ok(
    src.includes("override.tensor_parallel"),
    "reference knob vanished; this test needs re-anchoring",
  );
  assert.ok(
    src.includes("override.disable_vision"),
    "describeOverride does not mention disable_vision, so a Vision-off override is invisible in the API monitor",
  );
});
