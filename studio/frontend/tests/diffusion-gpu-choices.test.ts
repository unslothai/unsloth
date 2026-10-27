// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  type SystemGpuDevice,
  pinnableGpuContext,
} from "../src/hooks/gpu-selection.ts";

function device(
  index: number,
  overrides: Partial<SystemGpuDevice> = {},
): SystemGpuDevice {
  return {
    index,
    indexKind: "physical",
    name: `GPU ${index}`,
    memoryTotalGb: 24,
    memoryFreeGb: 20,
    sharedMemory: false,
    pinnable: true,
    diffusionPinnable: true,
    ...overrides,
  };
}

function choices(devices: SystemGpuDevice[] | null): SystemGpuDevice[] {
  const context = pinnableGpuContext(devices, true);
  return (context.ids?.length ?? 0) > 1 ? (context.devices ?? []) : [];
}

test("one card is nothing to choose between, so no control is offered", () => {
  assert.deepEqual(choices([device(0)]), []);
  assert.deepEqual(choices([]), []);
  assert.deepEqual(choices(null), []);
});

test("two pinnable cards are offered, in inventory order", () => {
  const offered = choices([device(0), device(1)]);
  assert.deepEqual(
    offered.map((d) => d.index),
    [0, 1],
  );
});

test("a masked host offers the physical ids it actually sees, not 0..n", () => {
  // Under CUDA_VISIBLE_DEVICES the backend reports physical ids and the routes translate them.
  const offered = choices([device(4), device(5)]);
  assert.deepEqual(
    offered.map((d) => d.index),
    [4, 5],
  );
});

test("cards the diffusion runner cannot address are not offered", () => {
  assert.deepEqual(
    choices([
      device(0, { diffusionPinnable: false }),
      device(1, { diffusionPinnable: false }),
    ]),
    [],
  );
  assert.deepEqual(choices([device(0), device(1, { diffusionPinnable: false })]), []);
});

test("the chat picker and the diffusion control answer independently", () => {
  const devices = [
    device(0, { indexKind: "vulkan", pinnable: true, diffusionPinnable: false }),
    device(1, { indexKind: "vulkan", pinnable: true, diffusionPinnable: false }),
  ];
  assert.deepEqual(pinnableGpuContext(devices, false).ids, [0, 1]);
  assert.deepEqual(choices(devices), []);
});

test("mixed index namespaces are never offered as one pool", () => {
  const devices = [
    device(0, { indexKind: "physical" }),
    device(1, { indexKind: "vulkan" }),
  ];
  assert.deepEqual(pinnableGpuContext(devices, true).ids, []);
  assert.deepEqual(choices(devices), []);
});

test("a pick whose card has gone falls back to automatic rather than a refusal", () => {
  const offered = choices([device(0), device(1)]);
  const send = (selected: string) =>
    selected !== "auto" && offered.some((d) => String(d.index) === selected)
      ? [Number(selected)]
      : undefined;
  assert.deepEqual(send("1"), [1]);
  assert.equal(send("auto"), undefined);
  assert.equal(send("7"), undefined);
});
