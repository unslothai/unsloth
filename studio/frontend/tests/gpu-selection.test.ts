// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  type GpuIndexKind,
  reconcileGpuSelection,
  resolveGpuSelectionContext,
  sameGpuSelection,
} from "../src/hooks/gpu-selection.ts";

const ids = [0, 1];

function reconcile(
  savedIndexKind: GpuIndexKind | null | undefined,
  currentIndexKind: GpuIndexKind | null | undefined,
  pinnableIds: number[] | null,
) {
  return reconcileGpuSelection(
    ids,
    savedIndexKind,
    currentIndexKind,
    pinnableIds,
  );
}

test("preserves every namespace while the device cache is genuinely cold", () => {
  assert.deepEqual(reconcile(undefined, undefined, null), {
    ids,
    indexKind: "physical",
  });
  assert.deepEqual(reconcile(null, undefined, null), {
    ids,
    indexKind: null,
  });
  assert.deepEqual(reconcile("physical", undefined, null), {
    ids,
    indexKind: "physical",
  });
  assert.deepEqual(reconcile("vulkan", undefined, null), {
    ids,
    indexKind: "vulkan",
  });
});

test("known unavailable Vulkan clears picks without a compatible namespace", () => {
  const context = resolveGpuSelectionContext([], false, "vulkan");
  assert.deepEqual(context, {
    devices: [],
    ids: null,
    indexKind: "vulkan",
  });
  assert.deepEqual(reconcile(undefined, context.indexKind, context.ids), {
    ids: null,
    indexKind: null,
  });
  assert.deepEqual(reconcile("physical", context.indexKind, context.ids), {
    ids: null,
    indexKind: null,
  });
  assert.deepEqual(reconcile(null, context.indexKind, context.ids), {
    ids: null,
    indexKind: null,
  });
  assert.deepEqual(reconcile("vulkan", context.indexKind, context.ids), {
    ids,
    indexKind: "vulkan",
  });
});

test("known Vulkan suppresses every DiffusionGemma pin while unavailable", () => {
  const context = resolveGpuSelectionContext([], true, "vulkan");
  assert.deepEqual(context, {
    devices: [],
    ids: [],
    indexKind: null,
  });
  for (const saved of [undefined, null, "physical", "vulkan"] as const) {
    assert.deepEqual(reconcile(saved, context.indexKind, context.ids), {
      ids: null,
      indexKind: null,
    });
  }
});

test("recovered inventories filter matching picks and reject mismatched namespaces", () => {
  assert.deepEqual(reconcile(null, "vulkan", [1]), {
    ids: null,
    indexKind: null,
  });
  assert.deepEqual(reconcile(null, "physical", [0, 1]), {
    ids: null,
    indexKind: null,
  });
  assert.deepEqual(reconcile("vulkan", "vulkan", [0]), {
    ids: [0],
    indexKind: "vulkan",
  });
  assert.deepEqual(reconcile(undefined, "vulkan", [0, 1]), {
    ids: null,
    indexKind: null,
  });
  assert.deepEqual(reconcile("physical", "physical", [1]), {
    ids: [1],
    indexKind: "physical",
  });
  assert.deepEqual(reconcile("vulkan", "physical", [0, 1]), {
    ids: null,
    indexKind: null,
  });
});

test("selection equality includes the GPU index namespace", () => {
  assert.equal(
    sameGpuSelection(
      { ids: [0, 1], indexKind: null },
      { ids: [0, 1], indexKind: "vulkan" },
    ),
    false,
  );
  assert.equal(
    sameGpuSelection(
      { ids: [0, 1], indexKind: "vulkan" },
      { ids: [0, 1], indexKind: "vulkan" },
    ),
    true,
  );
  assert.equal(
    sameGpuSelection(
      { ids: [0], indexKind: "physical" },
      { ids: [1], indexKind: "physical" },
    ),
    false,
  );
});
