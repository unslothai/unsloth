// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// When /api/inference/status may write the Vision switch. The control is what the
// next load or Apply sends; the loaded baseline is what the resident server is
// running. They are seeded on different rules, which is what lets them diverge.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { shouldSeedVisionSwitch } = await import(
  "../src/features/chat/lib/resolve-vision-switch-seed.ts"
);

const HERE = path.dirname(fileURLToPath(import.meta.url));
const APPLIER = readFileSync(
  path.join(
    HERE,
    "..",
    "src/features/chat/lib/apply-inference-status-to-store.ts",
  ),
  "utf8",
);

function seed(
  incoming: boolean,
  previous: {
    disableVision: boolean;
    loadedDisableVision: boolean | null;
    loadedVisionDisabledByUser: boolean | null;
  },
  hydratingExistingModel = false,
): boolean {
  return shouldSeedVisionSwitch({ incoming, previous, hydratingExistingModel });
}

test("an unseeded pair takes the running value", () => {
  assert.equal(
    seed(true, {
      disableVision: false,
      loadedDisableVision: null,
      loadedVisionDisabledByUser: null,
    }),
    true,
  );
});

test("a model or variant change reseeds the control it just left behind", () => {
  assert.equal(
    seed(
      true,
      {
        disableVision: false,
        loadedDisableVision: false,
        loadedVisionDisabledByUser: false,
      },
      true,
    ),
    true,
  );
});

test("a steady poll that agrees with the baseline writes nothing", () => {
  assert.equal(
    seed(false, {
      disableVision: true,
      loadedDisableVision: false,
      loadedVisionDisabledByUser: false,
    }),
    false,
  );
});

test("a poll that agrees with the baseline is not a reseed", () => {
  // Nothing moved and nothing is pending, so the answer is "no reseed needed".
  // Writing the incoming value anyway would be a no-op in the store, which is why
  // this is asserted on the resolver's answer rather than on a store effect: the
  // contract is what a future caller branches on.
  assert.equal(
    seed(false, {
      disableVision: false,
      loadedDisableVision: false,
      loadedVisionDisabledByUser: false,
    }),
    false,
  );
});

test("an external reload of the same model resyncs the control", () => {
  // Another tab or an API client loaded this model with the opposite setting.
  // loadedDisableVision and the image gate follow it unguarded, so leaving the
  // control behind shows Advanced Settings the opposite of the running projector
  // and arms the next Apply to undo the external change.
  assert.equal(
    seed(true, {
      disableVision: false,
      loadedDisableVision: false,
      loadedVisionDisabledByUser: false,
    }),
    true,
  );
});

test("a pending local edit survives an external reload", () => {
  // The control has been moved off its baseline and not applied yet. That is the
  // user's unapplied intent and a poll must not overwrite it, even though the
  // running server moved underneath.
  assert.equal(
    seed(true, {
      disableVision: true,
      loadedDisableVision: false,
      loadedVisionDisabledByUser: false,
    }),
    false,
  );
});

test("a seeded pair with no baseline yet is left alone", () => {
  // Nothing to compare against, so there is no evidence the server moved.
  assert.equal(
    seed(true, {
      disableVision: false,
      loadedDisableVision: null,
      loadedVisionDisabledByUser: false,
    }),
    false,
  );
});

test("the applier routes the control through the resolver", () => {
  // Re-inlining the old `loadedVisionDisabledByUser === null` guard would restore
  // the divergence without failing any of the cases above.
  assert.match(APPLIER, /shouldSeedVisionSwitch\(\{/);
  assert.doesNotMatch(
    APPLIER,
    /\(prevState\.loadedVisionDisabledByUser === null \|\|\s*\n?\s*hydratingExistingModel\) && \{\s*\n\s*disableVision:/,
  );
});
