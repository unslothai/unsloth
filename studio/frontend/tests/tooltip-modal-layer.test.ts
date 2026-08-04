// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { isTooltipLayerBlocked } from "../src/components/ui/tooltip-modal-layer.ts";

function tooltipLayer(pointerEvents: string): HTMLElement {
  return {
    ownerDocument: {
      defaultView: {
        getComputedStyle: () => ({ pointerEvents }),
      },
    },
  } as unknown as HTMLElement;
}

test("a tooltip below the active modal is blocked", () => {
  assert.equal(isTooltipLayerBlocked(tooltipLayer("none")), true);
});

test("a tooltip inside the active modal remains available", () => {
  assert.equal(isTooltipLayerBlocked(tooltipLayer("auto")), false);
});
