// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The GPUs list is an ordered list, not a set: position decides which card the
// model is given first. These cover the reordering the picker performs on it.

import assert from "node:assert/strict";
import test from "node:test";
import {
  reconcileGpuSelection,
  sameGpuSelection,
} from "../src/hooks/gpu-selection.ts";

import { readSrc } from "./helpers/kit.ts";

/** The picker's toggle, as model-config-page drives it. */
function toggle(current: number[], index: number): number[] {
  const next = current.includes(index)
    ? current.filter((i) => i !== index)
    : [...current, index];
  return next.length === 0 ? current : next;
}

/** The picker's move-earlier / move-later, as model-config-page drives it. */
function move(current: number[], index: number, delta: -1 | 1): number[] {
  const from = current.indexOf(index);
  const to = from + delta;
  if (from < 0 || to < 0 || to >= current.length) return current;
  const next = [...current];
  [next[from], next[to]] = [next[to], next[from]];
  return next;
}

test("moving a GPU earlier puts it in front of the model", () => {
  assert.deepEqual(move([0, 1], 1, -1), [1, 0]);
  assert.deepEqual(move([0, 1, 2], 2, -1), [0, 2, 1]);
});

test("the ends do not wrap", () => {
  assert.deepEqual(move([0, 1], 0, -1), [0, 1]);
  assert.deepEqual(move([0, 1], 1, 1), [0, 1]);
});

test("a GPU switched back on goes last, not back to its numeric slot", () => {
  // Re-inserting by index would silently undo a reorder the user just made.
  const afterDrop = toggle([1, 0], 1);
  assert.deepEqual(afterDrop, [0]);
  assert.deepEqual(toggle(afterDrop, 1), [0, 1]);
});

test("the last GPU cannot be switched off", () => {
  assert.deepEqual(toggle([0], 0), [0]);
});

test("the same set in a different order is a different placement", () => {
  const physical = (ids: number[] | null) =>
    ({ ids, indexKind: "physical" }) as const;
  assert.equal(sameGpuSelection(physical([0, 1]), physical([1, 0])), false);
  assert.equal(sameGpuSelection(physical([0, 1]), physical([0, 1])), true);
});

test("dropping an unpinnable GPU keeps the order of the rest", () => {
  assert.deepEqual(
    reconcileGpuSelection([2, 0, 1], "physical", "physical", [0, 1]).ids,
    [0, 1],
  );
  assert.deepEqual(
    reconcileGpuSelection([1, 0], "physical", "physical", [0, 1]).ids,
    [1, 0],
  );
});

// Diffusion drives ONE device and matches_gpu_ids reduces the request to its lowest
// id, so an ordering control there moves a row without moving the model. The arrows
// and the sentence promising the first card takes the prompt are both withheld.
test("the ordering controls and their promise are withheld for diffusion", () => {
  const src = readSrc("features/model-picker/components/model-config-page.tsx");
  const arrows = src.slice(src.indexOf("Move GPU ${d.index} earlier") - 900);
  assert.match(
    arrows.slice(0, 900),
    /!singleGpuInUse && !isDiffusion/,
    "the arrows must not render for a diffusion model",
  );
  assert.match(
    src,
    /!isDiffusion &&\s*" Their order here is the order they are given to the model/,
    "the help text must not promise ordering for a diffusion model",
  );
});

// The arrows move the list, but Apply is gated on the config differing from the
// loaded baseline. A sorting signature made a reorder read as no change, so the
// control moved the GPUs and could never apply them.
test("a reorder is a config change, so Apply stays reachable", async () => {
  const { gpuFieldsSignature } = await import(
    "../src/features/model-picker/model-config/config-signature.ts"
  );
  const base = { selectedGpuIds: [0, 1], selectedGpuIndexKind: "physical" } as never;
  const swapped = { selectedGpuIds: [1, 0], selectedGpuIndexKind: "physical" } as never;
  assert.notEqual(
    gpuFieldsSignature(base),
    gpuFieldsSignature(swapped),
    "a reordered pick must not share a signature with the loaded baseline",
  );
  const same = { selectedGpuIds: [0, 1], selectedGpuIndexKind: "physical" } as never;
  assert.equal(gpuFieldsSignature(base), gpuFieldsSignature(same));
});
