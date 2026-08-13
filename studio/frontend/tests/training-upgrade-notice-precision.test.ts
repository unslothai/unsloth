// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The Configure preview reports one precision for a run whose precision depends on which
// action the user picks at Start. A model that ships its own modeling code AND is shipped
// by the offered release gets both actions in the dialog: keeping the custom code loads it
// on the current transformers in 4-bit, and Install activates the latest sidecar, which
// trains 16-bit. The backend collapses that to forces_16bit=false (it answers for the
// fallback), so the card offered the install and went on promising QLoRA - 4-bit for a run
// that Install turns into a roughly threefold VRAM increase, or an OOM.

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

import type { TransformersUpgradeCheck } from "../src/features/transformers-upgrade/types.ts";

register("./helpers/transformers-upgrade-resolver.mjs", import.meta.url);

const { trainingTransformersUpgradeNotice } = await import(
  "../src/features/training/lib/training-transformers-upgrade.ts"
);

const RELEASE = "5.15.0";

/** Custom code AND an installable release: both dialog actions are selectable. */
const BOTH_ACTIONS: TransformersUpgradeCheck = {
  upgrade: {
    // biome-ignore lint/style/useNamingConvention: API schema
    model_type: "muse_glimmer",
    // biome-ignore lint/style/useNamingConvention: API schema
    pypi_version: RELEASE,
    // biome-ignore lint/style/useNamingConvention: API schema
    supported_in_pypi: true,
    // biome-ignore lint/style/useNamingConvention: API schema
    supported_in_main: true,
  },
  requiresTrustRemoteCode: true,
  latestTierActive: false,
  // The backend answers for the custom-code fallback, which keeps 4-bit.
  forces16Bit: false,
  installBreaksExactResume: false,
};

test("an offered install that is not already 16-bit discloses that Install switches to 16-bit", () => {
  const notice = trainingTransformersUpgradeNotice(BOTH_ACTIONS, true);
  assert.equal(notice.installVersion, RELEASE);
  // Not already 16-bit: picking the custom-code fallback really does keep 4-bit.
  assert.equal(notice.fourBitUnavailable, false);
  // But Install is selectable and turns this run 16-bit, so the card must say so.
  assert.equal(notice.installSwitchesTo16Bit, true);
});

test("a 16-bit run is not told twice that it will be 16-bit", () => {
  // No custom code: install_only_upgrade makes the backend answer forces_16bit itself,
  // and the existing warning already covers it.
  const notice = trainingTransformersUpgradeNotice(
    { ...BOTH_ACTIONS, requiresTrustRemoteCode: false, forces16Bit: true },
    true,
  );
  assert.equal(notice.fourBitUnavailable, true);
  assert.equal(notice.installSwitchesTo16Bit, false);
});

test("a run that never asked for 4-bit has no precision to lose", () => {
  // LoRA at 16-bit already: switching the overlay changes nothing it was promised.
  const notice = trainingTransformersUpgradeNotice(BOTH_ACTIONS, false);
  assert.equal(notice.installVersion, RELEASE);
  assert.equal(notice.fourBitUnavailable, false);
  assert.equal(notice.installSwitchesTo16Bit, false);
});

test("a dev-only upgrade has no Install action to warn about", () => {
  const notice = trainingTransformersUpgradeNotice(
    {
      ...BOTH_ACTIONS,
      upgrade: {
        // biome-ignore lint/style/useNamingConvention: API schema
        model_type: "muse_glimmer",
        // biome-ignore lint/style/useNamingConvention: API schema
        pypi_version: RELEASE,
        // biome-ignore lint/style/useNamingConvention: API schema
        supported_in_pypi: false,
        // biome-ignore lint/style/useNamingConvention: API schema
        supported_in_main: true,
      },
    },
    true,
  );
  assert.equal(notice.installVersion, null);
  assert.equal(notice.installSwitchesTo16Bit, false);
});

test("the sidecar already routing this model needs no action-dependent wording", () => {
  // Installed: there is no choice left to disclose, only the 16-bit fact.
  const notice = trainingTransformersUpgradeNotice(
    {
      upgrade: null,
      requiresTrustRemoteCode: true,
      latestTierActive: true,
      forces16Bit: true,
      installBreaksExactResume: false,
    },
    true,
  );
  assert.equal(notice.installVersion, null);
  assert.equal(notice.fourBitUnavailable, true);
  assert.equal(notice.installSwitchesTo16Bit, false);
});
