// SPDX-License-Identifier: AGPL-3.0-only

import assert from "node:assert/strict";
import test from "node:test";

import { installLocalStorageFake, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
installLocalStorageFake();
const { PLUS_MENU_ORDER, keepSavedSlot } =
  await import("../src/features/chat/stores/plus-menu-prefs-store.ts");

test("fixed-slot moves are repeatable without mutating readonly orders", () => {
  const fixed = "deepResearch";
  const movable = PLUS_MENU_ORDER.filter((id) => id !== fixed);

  for (let slot = 0; slot < PLUS_MENU_ORDER.length; slot += 1) {
    const saved: typeof PLUS_MENU_ORDER = [...movable];
    saved.splice(slot, 0, fixed);
    const savedBefore = [...saved];
    Object.freeze(saved);

    for (let rotation = 0; rotation < movable.length; rotation += 1) {
      const moved = [...movable.slice(rotation), ...movable.slice(0, rotation)];
      const movedBefore = [...moved];
      Object.freeze(moved);

      const result = keepSavedSlot(moved, saved, fixed);
      assert.equal(result.indexOf(fixed), slot);
      assert.equal(result.filter((id) => id === fixed).length, 1);
      assert.deepEqual(result.filter((id) => id !== fixed), movedBefore);
      Object.freeze(result);
      assert.deepEqual(keepSavedSlot(result, saved, fixed), result);
      assert.deepEqual(saved, savedBefore);
      assert.deepEqual(moved, movedBefore);
    }
  }
});
