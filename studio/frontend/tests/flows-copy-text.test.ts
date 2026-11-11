// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A card's copy button copies its first text output.

import assert from "node:assert/strict";
import test from "node:test";

import { outputCopyText } from "../src/features/flows/editor/value-kinds.ts";

test("a card copies its first text output, skipping media", () => {
  const ports = [{ id: "image" }, { id: "text" }];
  const image = { kind: "image", ref: "sha:abc" };
  assert.equal(outputCopyText(ports, { image, text: "a sloth on the moon" }), "a sloth on the moon");
  assert.equal(outputCopyText(ports, { image }), null);
  assert.equal(outputCopyText([{ id: "json" }], { json: { a: 1 } }), '{\n  "a": 1\n}');
  assert.equal(outputCopyText([{ id: "items" }], { items: ["x", { y: 2 }] }), 'x\n{\n  "y": 2\n}');
  assert.equal(outputCopyText([{ id: "n" }], { n: 3 }), "3");
  assert.equal(outputCopyText([], { text: "no ports" }), "no ports");
  assert.equal(outputCopyText([{ id: "text" }], null), null);
});
