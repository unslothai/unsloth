// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import assert from "node:assert/strict";
import test from "node:test";
import { readSrc } from "./helpers/kit.ts";

const runtime = readSrc("features/chat/hooks/use-chat-model-runtime.ts");
const slice = (from: string, to: string) => {
  const start = runtime.indexOf(from);
  const end = runtime.indexOf(to, start);
  assert.ok(start >= 0 && end > start, from);
  return runtime.slice(start, end);
};
const pick = new Function(
  "pendingLoadConfig",
  "stateBeforeUnload",
  "targetIsDiffusion",
  "reconcilePersistedGpuIds",
  "resetsPerModelSettings",
  `${slice("const stagedGpuIds =", "let loadSpeculativeType")}
  ${slice("const validateGpuIds =", "// The reset below")}
  if (resetsPerModelSettings) {
    ${slice("loadSelectedGpuIds = stagedGpuIds;", "loadGpuLayers =")}
  }
  return { validateGpuIds, loadSelectedGpuIds };`,
) as (...args: unknown[]) => { validateGpuIds: unknown; loadSelectedGpuIds: unknown };
const reconcile = (ids: number[] | null) => ids;

test("a switch validates the GPUs the load will use", () => {
  const cases: [unknown, boolean][] = [
    [{ selectedGpuIds: [1] }, true],
    [{ selectedGpuIds: null }, true],
    [undefined, true],
    [{ selectedGpuIds: [1] }, false],
    [undefined, false],
  ];
  for (const [pending, resets] of cases) {
    const { validateGpuIds, loadSelectedGpuIds } = pick(
      pending,
      { selectedGpuIds: [3], selectedGpuIndexKind: null },
      false,
      reconcile,
      resets,
    );
    assert.deepEqual(validateGpuIds, loadSelectedGpuIds, JSON.stringify([pending, resets]));
  }
});
