// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const read = (name: string) =>
  readFileSync(
    new URL(`../src/features/flows/player/world/${name}`, import.meta.url),
    "utf8",
  );

/** Required (non-optional) method names of the WorldGameHandle interface. */
function handleMethods(src: string): string[] {
  const body = src.slice(
    src.indexOf("export interface WorldGameHandle {"),
    src.indexOf("\n}", src.indexOf("export interface WorldGameHandle {")),
  );
  return [...body.matchAll(/^\s+([a-zA-Z]+)\(/gm)].map((m) => m[1]);
}

test("the 2D and 3D renderers both implement every game handle method", () => {
  const phaser = read("phaser-game.ts");
  const methods = handleMethods(phaser);
  assert.ok(methods.includes("addNpc") && methods.includes("addMarker"));
  for (const [name, src] of [
    ["phaser-game.ts", phaser],
    ["three-world.ts", read("three-world.ts")],
  ] as const) {
    // The handle object is the last object literal the factory returns.
    const handle = src.slice(src.lastIndexOf("\n  return {"));
    for (const m of methods)
      assert.match(handle, new RegExp(`\\n    ${m}\\(`), `${name} misses ${m}`);
  }
});
