// SPDX-License-Identifier: Apache-2.0
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

// #9244: the sidebar "More" menu's trigger drives open state in JS after
// preventDefault, so the modal dialog's body-level pointer-events:none shield
// never stops it. modalLayerActive() is the explicit consult of that shield.

import assert from "node:assert/strict";
import test from "node:test";

import { modalLayerActive } from "../src/lib/modal-layer-active.ts";

type FakeDoc = {
  body: object;
  defaultView: { getComputedStyle(body: object): { pointerEvents: string } };
};

function fakeDoc(pointerEvents: string): FakeDoc {
  const body = {};
  return {
    body,
    defaultView: { getComputedStyle: () => ({ pointerEvents }) },
  };
}

test("detects an active modal layer (pointer-events: none)", () => {
  assert.equal(modalLayerActive(fakeDoc("none") as unknown as Document), true);
});

test("a non-modal page reports no layer", () => {
  assert.equal(modalLayerActive(fakeDoc("auto") as unknown as Document), false);
  assert.equal(modalLayerActive(fakeDoc("") as unknown as Document), false);
});

test("a body-less document reports no layer rather than throwing", () => {
  const bodyless = { body: null, defaultView: null } as unknown as Document;
  assert.equal(modalLayerActive(bodyless), false);
});
