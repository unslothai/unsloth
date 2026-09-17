// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The consent dialog offered "Continue with custom code" only with nothing to install,
// or after an install failed. Training now raises it before a run starts, and for a
// model shipping its own modeling code that hid the only path still loading bnb 4-bit,
// since installing activates the 16-bit sidecar. So the first dialog offers the fallback
// next to Install.

import assert from "node:assert/strict";
import test from "node:test";

import { upgradeDialogActions } from "../src/features/transformers-upgrade/lib/upgrade-dialog-actions.ts";
import type { TransformersUpgradeInfo } from "../src/features/transformers-upgrade/types.ts";

const INSTALLABLE: TransformersUpgradeInfo = {
  // biome-ignore lint/style/useNamingConvention: API schema
  model_type: "muse_glimmer",
  // biome-ignore lint/style/useNamingConvention: API schema
  pypi_version: "5.15.0",
  // biome-ignore lint/style/useNamingConvention: API schema
  supported_in_pypi: true,
  // biome-ignore lint/style/useNamingConvention: API schema
  supported_in_main: true,
};
const DEV_ONLY: TransformersUpgradeInfo = {
  ...INSTALLABLE,
  // biome-ignore lint/style/useNamingConvention: API schema
  supported_in_pypi: false,
};

test("an installable release offers the custom-code way out alongside Install", () => {
  const actions = upgradeDialogActions({
    upgrade: INSTALLABLE,
    phase: "consent",
    trustRemoteCodeFallback: true,
  });

  assert.equal(actions.installable, true);
  assert.equal(actions.devOnly, false);
  assert.equal(
    actions.customCode,
    true,
    "installing forces 16-bit, so hiding this leaves a 4-bit run no way to start",
  );
});

test("a model without its own code still offers Install alone", () => {
  const actions = upgradeDialogActions({
    upgrade: INSTALLABLE,
    phase: "consent",
    trustRemoteCodeFallback: false,
  });

  assert.equal(actions.installable, true);
  assert.equal(actions.customCode, false);
});

test("a failed install keeps the fallback offered", () => {
  const actions = upgradeDialogActions({
    upgrade: INSTALLABLE,
    phase: "error",
    trustRemoteCodeFallback: true,
  });

  assert.equal(actions.customCode, true);
});

test("a running install offers nothing that would abandon it", () => {
  const actions = upgradeDialogActions({
    upgrade: INSTALLABLE,
    phase: "installing",
    trustRemoteCodeFallback: true,
  });

  assert.equal(actions.customCode, false);
});

test("a dev-only upgrade has nothing to install and says so", () => {
  const actions = upgradeDialogActions({
    upgrade: DEV_ONLY,
    phase: "consent",
    trustRemoteCodeFallback: true,
  });

  assert.equal(actions.installable, false);
  assert.equal(actions.devOnly, true);
  assert.equal(actions.customCode, true);
});

test("an upgrade no transformers ships at all offers only Cancel", () => {
  const actions = upgradeDialogActions({
    upgrade: {
      ...DEV_ONLY,
      // biome-ignore lint/style/useNamingConvention: API schema
      supported_in_main: false,
    },
    phase: "consent",
    trustRemoteCodeFallback: false,
  });

  assert.deepEqual(actions, {
    installable: false,
    devOnly: false,
    customCode: false,
  });
});
