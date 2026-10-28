// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const { mapRunConfigToOverride } =
  await import("../src/features/studio/sections/run-config-override.ts");

test("the run's objective comes from its saved config, before any reward is logged", () => {
  // Current Run mounts the sample answers card from this while the first batch is still scoring.
  assert.equal(
    mapRunConfigToOverride({ objective: "grpo" })?.objective,
    "grpo",
  );
  assert.equal(mapRunConfigToOverride({ objective: "dpo" })?.objective, "dpo");
  assert.equal(mapRunConfigToOverride({})?.objective, undefined);
});
