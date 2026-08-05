// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { isBlockedByActiveModal } from "../src/components/ui/tooltip-modal-layer.ts";

function element(pointerEvents: string): HTMLElement {
  return {
    ownerDocument: {
      defaultView: {
        getComputedStyle: () => ({ pointerEvents }),
      },
    },
  } as unknown as HTMLElement;
}

test("a trigger below the active modal is blocked", () => {
  assert.equal(isBlockedByActiveModal(element("none")), true);
});

test("a trigger inside the active modal remains available", () => {
  assert.equal(isBlockedByActiveModal(element("auto")), false);
});

test("a detached view answers no rather than throwing", () => {
  const orphan = {
    ownerDocument: { defaultView: null },
  } as unknown as HTMLElement;
  assert.equal(isBlockedByActiveModal(orphan), false);
});
