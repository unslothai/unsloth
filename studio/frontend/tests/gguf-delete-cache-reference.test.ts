// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import ts from "typescript";

function readSource(path: string) {
  const text = readFileSync(new URL(path, import.meta.url), "utf8");
  return { text, source: ts.createSourceFile(path, text, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX) };
}

const card = readSource("../src/features/hub/catalog/gguf-download-card.tsx");
const inventoryApi = readSource("../src/features/hub/inventory/api.ts");

test("the delete action forwards the redacted cache reference first", () => {
  // Redaction nulls `cache_path` and answers with `cache_ref`; a delete that reads only the
  // path sends nothing and the server reselects whichever duplicate ranks first.
  const deleteCall = card.text.slice(card.text.indexOf("await deleteCachedModel("));
  const args = deleteCall.slice(0, deleteCall.indexOf(");"));
  const referenceAt = args.indexOf("deleteTargetVariant?.cache_ref");
  const pathAt = args.indexOf("deleteTargetVariant?.cache_path");
  assert.ok(referenceAt > -1, "the delete call must read the row's cache_ref");
  assert.ok(pathAt > -1, "the path remains the fallback");
  assert.ok(referenceAt < pathAt, "cache_ref is preferred over the cleared cache_path");
  assert.match(args, /cachePath\s*\?\?/, "the card-level cachePath is still the last fallback");
});

test("the variant model declares cache_ref, so the reference survives parsing", () => {
  const declaration = inventoryApi.source.statements.find(
    (node) =>
      ts.isInterfaceDeclaration(node) && node.name.text === "GgufVariantDetail",
  );
  assert.ok(declaration, "GgufVariantDetail must exist");
  const fields = declaration.members
    .filter(ts.isPropertySignature)
    .map((member) => member.name.getText(inventoryApi.source));
  assert.ok(fields.includes("cache_ref"), `cache_ref missing from: ${fields.join(", ")}`);
});
