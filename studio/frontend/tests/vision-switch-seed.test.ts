// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The control is what the next load sends; the loaded baseline is what the server runs.
// They are seeded on different rules.

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
  // Asserted on the resolver's answer: the contract is what a caller branches on.
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
  // An external load with the opposite setting must move the control, or the next Apply undoes it.
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
  // Unapplied user intent must not be overwritten by a poll.
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
  // Re-inlining the old `loadedVisionDisabledByUser === null` guard would pass the cases above.
  assert.match(APPLIER, /shouldSeedVisionSwitch\(\{/);
  assert.doesNotMatch(
    APPLIER,
    /\(prevState\.loadedVisionDisabledByUser === null \|\|\s*\n?\s*hydratingExistingModel\) && \{\s*\n\s*disableVision:/,
  );
});
