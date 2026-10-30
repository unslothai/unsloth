// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Text outputs, long inputs and errors each have a copy button.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";
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

test("copy buttons sit on card outputs, inspector outputs, long inputs and errors", () => {
  const card = readSrc("features/flows/editor/flow-node.tsx");
  assert.match(card, /<CopyTextButton\s*text=\{copyText\}\s*label="Copy output"/);
  assert.match(card, /group-hover\/node:opacity-100/);
  const value = readSrc("features/flows/editor/output-value.tsx");
  assert.match(value, /<CopyTextButton text=\{value\}/);
  assert.match(value, /<CopyTextButton text=\{json\} label="Copy JSON"/);
  const nodeCard = readSrc("features/flows/editor/node-card.tsx");
  assert.match(nodeCard, /text=\{section\.text\}/);
  assert.match(nodeCard, /<CopyTextButton text=\{message\} label="Copy error"/);
  // The clamp is on the text, not the padded box, so no fourth line peeks through the padding.
  assert.match(nodeCard, /<p className="line-clamp-3 min-w-0 flex-1 text-ui-11 text-destructive">/);
  assert.match(readSrc("features/flows/editor/inspector.tsx"), /text=\{runtime\.error\} label="Copy error"/);
  const button = readSrc("features/flows/editor/copy-text.tsx");
  assert.match(button, /copyToClipboard\(text\)/);
  assert.match(button, /event\.stopPropagation\(\)/);
});

test("a card's text output leaves room on the right for its copy button", () => {
  const src = readSrc("features/flows/editor/flow-node.tsx");
  assert.match(src, /\$\{copyText \? " pr-9" : ""\}/);
});

test("number and toggle fields a doc never stored show the default the run uses", () => {
  const src = readSrc("features/flows/editor/param-fields.tsx");
  assert.match(src, /\(value === undefined \|\| value === null \? spec\.default : value\) \?\? ""/);
  assert.match(src, /checked=\{value === undefined \|\| value === null \? !!spec\.default : !!value\}/);
});
