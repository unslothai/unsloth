// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// "Keep multiple models loaded" is a three-way mode now: Off, When they fit,
// Always. An older backend still answers only {enabled}, so the read has to
// fall back to the bool, and the write sends the mode alone. The chat store
// keeps its bool, so the chat readers stay as they were.

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

import ts from "@typescript/typescript6";

import { installLocalStorageFake, readSrc } from "./helpers/kit.ts";
import { openingTag } from "./helpers/tsx-ast.ts";

register("./helpers/settings-api-resolver.mjs", import.meta.url);
installLocalStorageFake();

let nextBody: unknown = {};
const requests: { method: string; body: unknown }[] = [];

globalThis.fetch = (async (_input: RequestInfo | URL, init?: RequestInit) => {
  requests.push({
    method: init?.method ?? "GET",
    body: init?.body ? JSON.parse(String(init.body)) : undefined,
  });
  return new Response(JSON.stringify(nextBody), { status: 200 });
}) as typeof fetch;

const {
  isMultiModelMode,
  keepsModelsLoaded,
  loadMultiModelMode,
  normaliseMultiModelMode,
  updateMultiModelMode,
} = await import("../src/features/settings/api/multi-model.ts");

test("the mode is read tolerantly, and only Off unloads", () => {
  const cases: [unknown, string][] = [
    [{ enabled: false, mode: "always" }, "always"],
    [{ enabled: true, mode: "off" }, "off"],
    [{ enabled: true }, "fit"],
    // A mode this build does not know keeps the bool's answer, not "off".
    [{ enabled: true, mode: "bogus" }, "fit"],
    // biome-ignore lint/style/useNamingConvention: API schema
    [{ enabled: false, default_mode: "always" }, "off"],
    [{ enabled: "yes" }, "off"],
    [{}, "off"],
    [null, "off"],
  ];
  for (const [json, expected] of cases) {
    assert.equal(normaliseMultiModelMode(json), expected, JSON.stringify(json));
  }
  assert.deepEqual(
    ["off", "fit", "always", "Fit", "on", true].map(isMultiModelMode),
    [true, true, true, false, false, false],
  );
  assert.deepEqual((["off", "fit", "always"] as const).map(keepsModelsLoaded), [
    false,
    true,
    true,
  ]);
});

test("a save sends the mode alone; both reads normalise", async () => {
  requests.length = 0;
  nextBody = { enabled: true, mode: "fit" };
  assert.equal(await updateMultiModelMode("fit", "x"), "fit");
  assert.deepEqual(requests, [{ method: "PUT", body: { mode: "fit" } }]);
  nextBody = { enabled: true };
  assert.equal(await updateMultiModelMode("always", "x"), "fit");
  nextBody = { enabled: false, mode: "always" };
  assert.equal(await loadMultiModelMode("x"), "always");
});

// The row, read from the source: the node suite has no DOM.
test("the row is a three-item Select and the idle row closes the section", () => {
  const file = ts.createSourceFile(
    "model-memory-section.tsx",
    readSrc("features/settings/components/model-memory-section.tsx"),
    ts.ScriptTarget.ESNext,
    true,
    ts.ScriptKind.TSX,
  );
  const component = file.statements.find(
    (node): node is ts.FunctionDeclaration =>
      ts.isFunctionDeclaration(node) &&
      node.name?.getText() === "ModelMemorySection",
  );
  assert.ok(component, "ModelMemorySection is declared");
  const tags: ts.JsxOpeningLikeElement[] = [];
  const visit = (node: ts.Node): void => {
    const tag = openingTag(node);
    if (tag) {
      tags.push(tag);
    }
    ts.forEachChild(node, visit);
  };
  visit(component);
  const named = (name: string) =>
    tags.filter((tag) => tag.tagName.getText() === name);
  const attr = (tag: ts.JsxOpeningLikeElement, name: string) =>
    tag.attributes.properties
      .filter(ts.isJsxAttribute)
      .find((p) => p.name.getText() === name)
      ?.initializer?.getText();

  assert.equal(named("Select").length, 1);
  const [trigger] = named("SelectTrigger");
  assert.deepEqual(
    [attr(trigger, "size"), attr(trigger, "className")],
    ['"sm"', '"w-48"'],
  );
  assert.match(attr(trigger, "aria-label") ?? "", /modelMemory\.multiModel"/);
  assert.deepEqual(
    named("SelectItem").map((item) => attr(item, "value")),
    ['"off"', '"fit"', '"always"'],
  );
  for (const sw of named("Switch")) {
    assert.doesNotMatch(attr(sw, "aria-label") ?? "", /multiModel"/);
  }
  // The chat store bool follows the mode: after the read and after a save.
  assert.equal(
    component.getText().match(/setKeepModelsLoaded\(keepsModelsLoaded\(/g)
      ?.length,
    2,
  );

  const [section] = named("SettingsSection");
  assert.equal(attr(section, "ref"), "{ref}");
  const last = (section.parent as ts.JsxElement).children
    .filter((c) => !(ts.isJsxText(c) && c.containsOnlyTriviaWhiteSpaces))
    .at(-1);
  const lastTag = last ? openingTag(last) : null;
  assert.equal(lastTag?.tagName.getText(), "MediaIdleUnloadSetting");
  assert.match(
    lastTag ? (attr(lastTag, "residencyKey") ?? "") : "",
    /keepResident/,
  );
});
