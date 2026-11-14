// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  MLX_OFFLOAD_SLIDER_AUTO,
  mlxOffloadDecided,
  mlxOffloadFromSlider,
  mlxOffloadModePatch,
  mlxOffloadSliderValue,
} from "../src/features/model-picker/model-config/memory-fit.ts";

test("Manual pins what Auto would pick, and off where it keeps a part resident", () => {
  const parts = [
    { key: "mlxExpertOffload", decided: 0.6 },
    { key: "mlxPleOffload", decided: null },
  ] as const;
  assert.deepEqual(mlxOffloadModePatch(true, parts), { mlxExpertOffload: 0.6, mlxPleOffload: "off" });
  assert.deepEqual(mlxOffloadModePatch(true, [{ key: "mlxPleOffload", decided: 0 }]), {
    mlxPleOffload: "off",
  });
});

test("Default leaves every shown part to Auto", () => {
  assert.deepEqual(
    mlxOffloadModePatch(false, [
      { key: "mlxExpertOffload", decided: 0.6 },
      { key: "mlxPleOffload", decided: 1 },
    ]),
    { mlxExpertOffload: null, mlxPleOffload: null },
  );
});

test("the slider has an Auto step below off, so a part on Auto is never shown as off", () => {
  assert.equal(mlxOffloadSliderValue(null), MLX_OFFLOAD_SLIDER_AUTO);
  assert.equal(mlxOffloadSliderValue("off"), 0);
  assert.equal(mlxOffloadSliderValue(0.25), 25);
  assert.deepEqual([MLX_OFFLOAD_SLIDER_AUTO, 0, 60].map(mlxOffloadFromSlider), [null, "off", 0.6]);
});

test("Auto's pick is read only from an estimate of the current settings", () => {
  const estimate = { mlxPleOffload: 1, mlxExpertOffload: 0.5 };
  assert.deepEqual(mlxOffloadDecided({ estimate, stale: false, loading: false }), { ple: 1, experts: 0.5 });
  // Unknown, not resident: Manual would otherwise pin every part off while the estimate refreshes.
  assert.equal(mlxOffloadDecided({ estimate, stale: true, loading: false }), null);
  assert.equal(mlxOffloadDecided({ estimate, stale: false, loading: true }), null);
  assert.equal(mlxOffloadDecided({ estimate: null, stale: false, loading: false }), null);
});
