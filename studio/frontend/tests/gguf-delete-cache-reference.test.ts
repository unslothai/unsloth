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
const impact = readSource("../src/features/hub/catalog/delete-impact.tsx");
const pickers = readSource(
  "../src/features/model-picker/components/model-selector/pickers.tsx",
);

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
    (node): node is ts.InterfaceDeclaration =>
      ts.isInterfaceDeclaration(node) && node.name.text === "GgufVariantDetail",
  );
  assert.ok(declaration, "GgufVariantDetail must exist");
  const fields = declaration.members
    .filter(ts.isPropertySignature)
    .map((member) => member.name.getText(inventoryApi.source));
  assert.ok(fields.includes("cache_ref"), `cache_ref missing from: ${fields.join(", ")}`);
});

test("a logical quant delete carries the copy its own row was listed in", () => {
  // The server re-resolves a quant-level delete that carries no copy, and that resolution
  // cannot see the scoped companion readiness the online listing ranked duplicates by, so the
  // row's own cache path is the only thing that keeps the delete on the advertised copy.
  const expanderCall = pickers.text.match(
    /await onDeleteVariant\(v\.quant, v\.cache_ref \?\? v\.cache_path\)/,
  );
  assert.ok(expanderCall, "the expander row must hand its listed copy to the delete");
  const impact = pickers.text.match(
    /impact: \{\s*repoId,\s*variant: v\.quant,\s*cachePath: v\.cache_ref \?\? v\.cache_path,\s*\}/,
  );
  assert.ok(impact, "the confirm preview must measure the same copy the delete removes");

  const pinnedCopy = pickers.text.match(
    /const pinnedCopyPath = downloadedPinnedQuantPaths\.get\(/,
  );
  assert.ok(pinnedCopy, "the pinned row must look up the copy its validation resolved");
  assert.match(
    pickers.text,
    /pinnedCopyPath \?\? undefined,/,
    "the pinned delete must forward that copy",
  );
  assert.match(
    pickers.text,
    /cachePath \?\?\n\s*\(mediaPageForTask\(c\.task\) \? c\.cache_path \|\| undefined : undefined\)/,
    "the On Device expander must prefer the row's copy and keep the media fallback",
  );
});

test("the delete preview accepts the copy so it measures that one", () => {
  const fn = inventoryApi.text.slice(
    inventoryApi.text.indexOf("export async function fetchDeleteImpact"),
  );
  const body = fn.slice(0, fn.indexOf("\n}"));
  assert.match(body, /cachePath\?: string \| null,/, "the preview must accept a copy");
  assert.ok(
    body.includes("cachePath ? { cache_path: cachePath } : {}"),
    "the preview must send the copy it was given",
  );
  assert.ok(
    impact.text.includes("cachePath?: string | null,") &&
      impact.text.includes("fetchDeleteImpact(repoId, variant ?? undefined, cachePath ?? undefined)"),
    "the hook must forward the copy and refetch when it changes",
  );
  const menu = readSource("../src/features/model-picker/components/model-selector/model-row-menu.tsx");
  assert.match(
    menu.text,
    /impact\?: \{ repoId: string; variant\?: string \| null; cachePath\?: string \| null \}/,
    "the row menu must accept the copy for its preview",
  );
  assert.match(
    menu.text,
    /del\?\.impact\?\.cachePath,/,
    "the row menu must pass the copy to the preview hook",
  );
});


test("a redacted listing keeps the copy on the picker rows and their previews", () => {
  // Redaction nulls `cache_path` and answers with `cache_ref`. A row that reads only the path
  // sends nothing, and the server then ranks duplicates itself, which cannot see the scoped
  // readiness the listing ranked them by.
  const chatTypes = readSource("../src/features/chat/types/api.ts");
  const declaration = chatTypes.source.statements.find(
    (node): node is ts.InterfaceDeclaration =>
      ts.isInterfaceDeclaration(node) && node.name.text === "GgufVariantDetail",
  );
  assert.ok(declaration, "the chat GgufVariantDetail must exist");
  const fields = declaration.members
    .filter(ts.isPropertySignature)
    .map((member) => member.name.getText(chatTypes.source));
  assert.ok(
    fields.includes("cache_ref"),
    `the chat variant model must declare cache_ref: ${fields.join(", ")}`,
  );

  // The picker rows: expander, sole quant, and the pinned copy the validation listing resolved.
  assert.match(
    pickers.text,
    /await onDeleteVariant\(v\.quant, v\.cache_ref \?\? v\.cache_path\)/,
    "the expander delete must prefer the reference",
  );
  assert.match(
    pickers.text,
    /cachePath: v\.cache_ref \?\? v\.cache_path,/,
    "the expander preview must measure the same copy",
  );
  assert.match(
    pickers.text,
    /pinKey\(repoId, variant\.quant\),\s*\n\s*variant\.cache_ref \?\? variant\.cache_path \?\? null,/,
    "the pinned copy lookup must prefer the reference",
  );
  assert.match(
    pickers.text,
    /variant\.cache_ref \?\?\n\s*variant\.cache_path \?\?\n\s*\(mediaPageForTask\(c\.task\) \? c\.cache_path : null\)/,
    "the On Device row preview must prefer the reference and keep the media fallback",
  );
  assert.match(
    pickers.text,
    /variant\.cache_ref \?\?\n\s*variant\.cache_path \?\?\n\s*\(mediaPageForTask\(c\.task\) \? c\.cache_path : undefined\)/,
    "the On Device delete must prefer the reference and keep the media fallback",
  );
});

test("the disk card previews the copy its delete sends", () => {
  // The delete already preferred the reference; the preview above it did not, so an API-key
  // caller was shown another duplicate's reclaimed bytes.
  const preview = card.text.slice(card.text.indexOf("const deleteImpact = useDeleteImpact("));
  const args = preview.slice(0, preview.indexOf(");"));
  assert.match(
    args,
    /deleteTargetVariant\?\.cache_ref \?\? deleteTargetVariant\?\.cache_path \?\? cachePath \?\? undefined/,
    "the preview must receive the same reference the delete sends",
  );
});
