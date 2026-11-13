// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import ts from "@typescript/typescript6";
import { readSrc } from "./helpers/kit.ts";

// Runs the real layout effect on stubbed DOM sizes, without React.
const source = ts.createSourceFile(
  "media-rail-resize-handle.tsx",
  readSrc("components/media-rail-resize-handle.tsx"),
  ts.ScriptTarget.Latest,
  true,
  ts.ScriptKind.TSX,
);
let effectSource = "";
function visit(node: ts.Node): void {
  if (ts.isCallExpression(node) && node.expression.getText(source) === "useLayoutEffect") {
    effectSource = node.arguments[0].getText(source);
  }
  ts.forEachChild(node, visit);
}
visit(source);
assert.ok(effectSource, "the rail cap must update from actual layout measurements");
const runEffect = new Function(
  "anchorRef", "placement", "rail", "kind", "getComputedStyle", "document", "ResizeObserver", "setCap",
  `return (${effectSource})();`,
);

function measure({
  kind = "images", scale = 1, inset = "", fallbackInset = "", placement = "page",
}: {
  kind?: "images" | "video" | "audio";
  scale?: number;
  inset?: string;
  fallbackInset?: string;
  placement?: "page" | "rail";
} = {}) {
  const block = { clientWidth: 1000 };
  const root = {};
  let cap = NaN;
  let update = () => {};
  let disconnected = false;
  const cleanup = runEffect(
    { current: { parentElement: placement === "page" ? block : { parentElement: block } } },
    placement, { scale }, kind,
    (element: unknown) => element === root ? { fontSize: "16px" } : {
      paddingLeft: "24px", paddingRight: "24px",
      getPropertyValue: (name: string) => name === "--studio-chat-header-right-inset" ? inset : fallbackInset,
    },
    { documentElement: root },
    class {
      constructor(callback: () => void) { update = callback; }
      observe(element: unknown) { assert.equal(element, block); }
      disconnect() { disconnected = true; }
    },
    (value: number) => { cap = value; },
  ) as () => void;
  return {
    get cap() { return cap; },
    resize(width: number) { block.clientWidth = width; update(); },
    cleanup() { cleanup(); assert.ok(disconnected); },
  };
}

test("image and video drag/keyboard caps reserve the scaled compact toolbar", () => {
  for (const kind of ["images", "video"] as const) {
    const normal = measure({ kind });
    assert.equal(normal.cap, 776);
    normal.cleanup();
    const enlarged = measure({ kind, scale: 2 });
    assert.equal(enlarged.cap, 276);
    enlarged.resize(1200);
    assert.equal(enlarged.cap, 376);
    enlarged.cleanup();
  }
});

test("the header inset takes precedence, including zero, and window chrome is not scaled", () => {
  const zero = measure({ inset: "0px", fallbackInset: "138px" });
  assert.equal(zero.cap, 776);
  zero.cleanup();
  const fallback = measure({ fallbackInset: "138px", scale: 2 });
  assert.equal(fallback.cap, 207);
  fallback.cleanup();
  const explicit = measure({ inset: "50px", fallbackInset: "138px", scale: 2 });
  assert.equal(explicit.cap, 251);
  explicit.cleanup();
});

test("audio retains its existing preview minimum and inset rail placement excludes padding", () => {
  const audio = measure({ kind: "audio", scale: 2, inset: "138px" });
  assert.equal(audio.cap, 396);
  audio.cleanup();
  const rail = measure({ placement: "rail" });
  assert.equal(rail.cap, 728);
  rail.cleanup();
});
