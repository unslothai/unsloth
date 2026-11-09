// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The API-only switch must render without chat idle unload, or the media TTL row is stuck paused.

import assert from "node:assert/strict";
import test from "node:test";

import ts from "@typescript/typescript6";

import { readSrc } from "./helpers/kit.ts";
import { openingTag } from "./helpers/tsx-ast.ts";

const SECTION = readSrc("features/settings/components/model-auto-switch-section.tsx");

const GUARD = "\n      {settings";

function guardFor(labelKey: string): string {
  const upTo = SECTION.slice(0, SECTION.indexOf(`modelAutoSwitch.${labelKey}"`));
  return upTo.slice(upTo.lastIndexOf(GUARD));
}

test("the API-only switch is reachable whenever a media TTL is saved", () => {
  assert.match(guardFor("apiOnly"), /mediaAutoUnloadIdleSeconds > 0/);
});

test("it is still reachable from the chat TTL alone", () => {
  assert.match(guardFor("apiOnly"), /idleUnloadActive/);
});

test("the KV-save option renders without the chat TTL", () => {
  // It also saves the chat KV before an image or video model takes the GPU, so no guard.
  const file = ts.createSourceFile("s.tsx", SECTION, ts.ScriptTarget.ESNext, true, ts.ScriptKind.TSX);
  let row: ts.Node | undefined;
  const visit = (node: ts.Node): void => {
    const tag = openingTag(node);
    if (tag?.tagName.getText() === "SettingsRow" && /modelAutoSwitch\.keepKv"/.test(tag.getText())) row = node;
    ts.forEachChild(node, visit);
  };
  visit(file);
  assert.ok(row, "the keepKv row is rendered");
  assert.doesNotMatch(row.getText(), /mediaAutoUnloadIdleSeconds/);
  for (let node = row.parent; openingTag(node)?.tagName.getText() !== "SettingsSection"; node = node.parent) {
    const guard = ts.isConditionalExpression(node) || (ts.isBinaryExpression(node) && node.operatorToken.kind === ts.SyntaxKind.AmpersandAmpersandToken);
    assert.ok(!guard, `keepKv is behind ${node.getText().slice(0, 40)}`);
  }
});

test("the media row still says when a veto is holding its TTL", () => {
  assert.match(
    SECTION,
    /settings\.mediaAutoUnloadIdleSeconds > 0 &&\s*!settings\.mediaIdleUnloadActive/,
  );
});
